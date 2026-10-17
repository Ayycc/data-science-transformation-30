"""End-to-end pipeline: data -> segmentation -> forecasts -> policies -> MILP -> SimPy comparison -> results/."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as data_mod
from .forecasting import final_forecasts, holdout_evaluation, pinball, quantile, select_models
from .milp import objective_value, solve_pulp
from .policy import costs_per_item, heuristic_sS, newsvendor, optimised_sS
from .segmentation import segment
from .simulation import SimInputs, SimResult, simulate


def _season_mask(cfg: dict, n_weeks: int) -> np.ndarray:
    d = cfg["data"]
    woy = np.arange(n_weeks) % 52
    return (woy >= d["season_start_week"]) & (woy < d["season_start_week"] + d["season_length_weeks"])


def prepare(cfg: dict, ds: data_mod.Dataset) -> dict:
    """Everything up to (and including) the policy parameters; no simulation."""
    H = int(cfg["data"]["history_weeks"])
    T = int(cfg["data"]["test_weeks"])
    hist = ds.history(H)
    mask = _season_mask(cfg, H + T)
    seg = segment(hist, ds.items, ds.skus, cfg, season_mask=mask)
    m = ds.items.set_index("item_id").join(ds.skus.set_index("sku_id"), on="sku_id").loc[seg.index]
    wh_names = [w["name"] for w in cfg["warehouses"]]
    m["wh_idx"] = m["warehouse"].map({n: i for i, n in enumerate(wh_names)})

    non_seasonal = seg.index[~seg["seasonal"]]
    seasonal = seg.index[seg["seasonal"]]
    per_item, fsum = holdout_evaluation(hist, seg, int(cfg["data"]["validation_weeks"]), non_seasonal)
    choice = select_models(fsum)
    fc = final_forecasts(hist, seg, choice, per_item)
    opt = optimised_sS(fc, m, seg, cfg)
    heur = heuristic_sS(hist, m, cfg, non_seasonal)
    nv = newsvendor(hist, m, seasonal, mask, cfg) if len(seasonal) else pd.DataFrame()
    test = ds.test(H).pivot(index="item_id", columns="week", values="demand").reindex(seg.index).fillna(0)
    return {
        "hist": hist,
        "seg": seg,
        "m": m,
        "per_item": per_item,
        "fsum": fsum,
        "choice": choice,
        "fc": fc,
        "opt": opt,
        "heur": heur,
        "nv": nv,
        "test": test,
        "mask": mask,
        "H": H,
        "T": T,
    }


def sim_inputs(cfg: dict, prep: dict, ds: data_mod.Dataset, policy: str, lt_cv_mult: float = 1.0) -> SimInputs:
    """Assemble arrays for one policy ("heuristic" or "optimised")."""
    seg, m, fc, T, H = prep["seg"], prep["m"], prep["fc"], prep["T"], prep["H"]
    idx = seg.index
    n = len(idx)
    seasonal = seg["seasonal"].to_numpy()
    k = costs_per_item(m, cfg)
    s = np.zeros(n)
    S = np.zeros(n)
    if policy == "optimised":
        pol = prep["opt"] if lt_cv_mult == 1.0 else optimised_sS(fc, m, seg, cfg, lt_cv_mult)
    else:
        pol = prep["heur"]
    pos = idx.get_indexer(pol.index)
    s[pos], S[pos] = pol["s"].to_numpy(), pol["S"].to_numpy()
    season_qty = np.zeros(n)
    order_week = np.full(n, -1)
    nv = prep["nv"]
    if len(nv):
        p2 = idx.get_indexer(nv.index)
        season_qty[p2] = nv["q_newsvendor" if policy == "optimised" else "q_heuristic"].to_numpy()
        ltm = m.loc[nv.index, "lt_mean_weeks"].to_numpy()
        ltcv = m.loc[nv.index, "lt_cv"].to_numpy()
        start = int(cfg["data"]["season_start_week"])
        order_week[p2] = np.maximum(start - np.ceil(ltm * (1 + 2 * ltcv)).astype(int) - 1, 0)
    mu = fc["mu"].reindex(idx).fillna(0.0).to_numpy()
    cost, vol = m["unit_cost"].to_numpy(), m["volume_m3"].to_numpy()
    wh = m["wh_idx"].to_numpy()
    mcfg = cfg["milp"]
    budget = float(mcfg["weekly_budget_multiplier"] * np.sum(cost * mu))
    season_mean = np.zeros(n)
    if len(nv):
        season_mean[idx.get_indexer(nv.index)] = nv["season_mean"].to_numpy()
    cap = np.bincount(
        wh, weights=mcfg["capacity_weeks_of_cover"] * vol * mu + vol * season_mean, minlength=len(cfg["warehouses"])
    )
    hist_mean = seg["mean_weekly"].to_numpy()
    init = np.where(seasonal, 0, np.ceil(hist_mean * (m["lt_mean_weeks"].to_numpy() + 1)))
    W = int(cfg["simulation"]["warmup_weeks"])
    lts = data_mod.lead_time_samples(
        ds.skus, ds.items.set_index("item_id").loc[idx].reset_index(), W + T, int(cfg["seed"]), lt_cv_mult
    )
    d = cfg["data"]
    warm = prep["hist"][prep["hist"]["week"] >= H - W].pivot(index="item_id", columns="week", values="demand").reindex(idx)
    demand = np.hstack([warm.fillna(0).to_numpy(), prep["test"].to_numpy()]).astype(float)
    order_week = np.where(order_week >= 0, order_week + W, -1)
    return SimInputs(
        demand=demand,
        warmup=W,
        lead_times=lts,
        s=s,
        S=S,
        pack=m["pack_size"].to_numpy(),
        moq=m["moq"].to_numpy(),
        cost=cost,
        price=m["price"].to_numpy(),
        vol=vol,
        wh=wh,
        h_week=k["h_week"].to_numpy(),
        shortage=k["shortage"].to_numpy(),
        K=float(cfg["costs"]["ordering_cost_per_line"]),
        budget=budget,
        capacity=cap,
        init_on_hand=init,
        seasonal=seasonal,
        season_qty=season_qty,
        season_order_week=order_week,
        season_end_week=W + int(d["season_start_week"] + d["season_length_weeks"] - 1),
        salvage=cost * cfg["costs"]["seasonal_salvage_frac_cost"],
        w_s=mcfg["below_s_weight"] * k["shortage"].to_numpy(),
        w_S=mcfg["below_S_weight"] * k["shortage"].to_numpy(),
    )


def summarise(res: SimResult, inp: SimInputs, seg: pd.DataFrame) -> dict:
    out = {
        "fill_rate": round(res.fill_rate, 4),
        "total_cost": round(res.total_cost, 0),
        "holding_cost": round(res.holding_cost, 0),
        "shortage_cost": round(res.shortage_cost, 0),
        "ordering_cost": round(res.ordering_cost, 0),
        "seasonal_overage_cost": round(res.overage_cost, 0),
        "avg_inventory_value": round(res.avg_inventory_value, 0),
        "purchase_spend": round(res.purchase_spend, 0),
        "order_lines": res.n_order_lines,
        "weeks_over_budget": int(np.sum(res.weekly_spend > inp.budget + 1e-6)),
        "weeks_budget_binding": int(np.sum(res.weekly_spend >= 0.98 * inp.budget)),
        "weeks_capacity_above_98pct": int(np.sum(res.weekly_capacity_util >= 0.98)),
        "mean_peak_capacity_util": round(float(res.weekly_capacity_util.mean()), 4),
    }
    val = res.avg_on_hand * inp.cost
    for cls in ("A", "B", "C"):
        mk = (seg["abc"] == cls).to_numpy() & ~inp.seasonal
        out[f"avg_inventory_value_{cls}"] = round(float(val[mk].sum()), 0)
    out["avg_inventory_value_seasonal"] = round(float(val[inp.seasonal].sum()), 0)
    for cls in ("A", "B", "C"):
        mk = (seg["abc"] == cls).to_numpy() & ~inp.seasonal
        out[f"fill_rate_{cls}"] = round(float(res.served[mk].sum() / max(res.demand[mk].sum(), 1)), 4)
    mk = inp.seasonal
    out["fill_rate_seasonal"] = round(float(res.served[mk].sum() / max(res.demand[mk].sum(), 1)), 4) if mk.any() else None
    return out


def quantile_backtest(prep: dict) -> dict:
    """Coverage and pinball loss of weekly quantile forecasts on the unseen test year (non-seasonal items)."""
    fc, test, hist = prep["fc"], prep["test"], prep["hist"]
    y = test.loc[fc.index].to_numpy()
    mu = np.repeat(fc["mu"].to_numpy()[:, None], y.shape[1], 1)
    sd = np.repeat(fc["sigma"].to_numpy()[:, None], y.shape[1], 1)
    dist = np.repeat(fc["dist"].to_numpy()[:, None], y.shape[1], 1)
    wide = hist.pivot(index="item_id", columns="week", values="demand").loc[fc.index].to_numpy()
    out = {}
    for q in (0.5, 0.9, 0.95):
        qh = quantile(mu, sd, q, dist)
        emp = np.repeat(np.quantile(wide[:, -13:], q, axis=1)[:, None], y.shape[1], 1)  # planner: recent 13-wk empirical
        out[f"p{int(q * 100)}"] = {
            "coverage": round(float(np.mean(y <= qh)), 4),
            "pinball": round(pinball(y, qh, q), 4),
            "pinball_empirical_13wk": round(pinball(y, emp, q), 4),
        }
    return out


def run(
    cfg: dict,
    results: Path,
    data_dir: Path,
    sample_dir: Path | None = None,
    n_skus: int | None = None,
    sensitivity: bool = True,
    figures: bool = True,
) -> dict:
    t0 = time.perf_counter()
    results.mkdir(parents=True, exist_ok=True)
    ok, msg = data_mod.probe_m5(cfg["data"]["m5_probe_url"], cfg["data"]["m5_probe_timeout_s"])
    ds = data_mod.generate(cfg, n_skus=n_skus)
    ds.source = "synthetic (M5 probe: " + ("reachable, adapter not implemented" if ok else "unreachable") + ")"
    data_mod.save(ds, data_dir)
    if sample_dir is not None:
        keep = ds.skus["sku_id"].iloc[:40]
        smp = data_mod.Dataset(
            skus=ds.skus[ds.skus["sku_id"].isin(keep)],
            items=ds.items[ds.items["sku_id"].isin(keep)],
            demand=ds.demand[ds.demand["item_id"].str.split("|").str[0].isin(keep)],
            source=ds.source,
        )
        data_mod.save(smp, sample_dir)

    prep = prepare(cfg, ds)
    seg = prep["seg"]
    t_prep = time.perf_counter() - t0

    runs = {}
    inputs = {}
    trace_items = np.array([0, 1, 2])
    for name, policy, alloc in (
        ("heuristic", "heuristic", "greedy"),
        ("optimised_greedy", "optimised", "greedy"),
        ("optimised_milp", "optimised", "milp"),
    ):
        inp = sim_inputs(cfg, prep, ds, policy)
        res = simulate(
            inp,
            allocator=alloc,
            time_limit_s=cfg["milp"]["time_limit_s"],
            rel_gap=cfg["milp"]["relative_gap"],
            trace_items=trace_items,
        )
        runs[name], inputs[name] = res, inp

    summ = {k: summarise(v, inputs[k], seg) for k, v in runs.items()}
    base, best = summ["heuristic"], summ["optimised_milp"]

    # MILP diagnostics over every weekly solve.
    st = pd.DataFrame(runs["optimised_milp"].milp_stats)
    milp_diag = {
        "weekly_solves": int(len(st)),
        "optimal": int((st["status"] == "OPTIMAL").sum()) if len(st) else 0,
        "max_gap": float(np.nanmax(st["gap"])) if len(st) else 0.0,
        "max_triggered_items": int(st["triggered"].max()) if len(st) else 0,
        "max_variables": int(st["n_vars"].max()) if len(st) else 0,
        "median_seconds": round(float(st["seconds"].median()), 3) if len(st) else 0.0,
        "max_seconds": round(float(st["seconds"].max()), 3) if len(st) else 0.0,
        "weeks_budget_binding": int((st["spend"] >= 0.98 * st["budget"]).sum()) if len(st) else 0,
    }
    # Cross-check: PuLP/CBC re-solves the first scored week's instance of the optimised policy.
    inst, ip_trig, trig, s_or = runs["optimised_milp"].first_instance
    s_pl = solve_pulp(inst, cfg["milp"]["crosscheck_time_limit_s"])
    milp_diag["week1_crosscheck"] = {
        "items": int(inst.n),
        "ortools_scip_status": s_or.status,
        "ortools_objective": round(s_or.objective, 2),
        "ortools_best_bound": round(s_or.bound, 2),
        "ortools_gap": s_or.gap,
        "pulp_cbc_status": s_pl.status,
        "pulp_objective": round(objective_value(inst, s_pl.q), 2),
        "budget": round(inst.budget, 2),
        "spend_ortools": round(float((s_or.q * inst.cost).sum()), 2),
    }
    week0 = pd.DataFrame(
        {
            "item_id": seg.index[trig],
            "s": inst.gap_s + ip_trig - 1,
            "S": inst.gap_S + ip_trig,
            "inventory_position": ip_trig,
            "moq": inst.moq,
            "pack": inst.pack,
            "order_qty": s_or.q,
            "spend": s_or.q * inst.cost,
        }
    )
    week0.to_csv(results / "replenishment_plan_week1.csv", index=False)

    # Sensitivity on lead-time variance (stretch goal).
    sens = []
    if sensitivity:
        for mult in cfg["sensitivity"]["lead_time_cv_multipliers"]:
            for name, policy, alloc in (("heuristic", "heuristic", "greedy"), ("optimised_milp", "optimised", "milp")):
                if mult == 1.0:
                    sm = summ[name]
                else:
                    i2 = sim_inputs(cfg, prep, ds, policy, lt_cv_mult=mult)
                    sm = summarise(
                        simulate(
                            i2, allocator=alloc, time_limit_s=cfg["milp"]["time_limit_s"], rel_gap=cfg["milp"]["relative_gap"]
                        ),
                        i2,
                        seg,
                    )
                sens.append({"lt_cv_multiplier": mult, "policy": name, **sm})
        pd.DataFrame(sens).to_csv(results / "sensitivity_lead_time_cv.csv", index=False)

    qb = quantile_backtest(prep)

    # Tables
    seg_tab = seg.groupby(["abc", "xyz"]).size().unstack(fill_value=0)
    seg_tab.to_csv(results / "abc_xyz_counts.csv")
    prep["fsum"].pivot(index="sbc", columns="model", values="smae").round(4).to_csv(results / "forecast_holdout_smae.csv")
    prep["fsum"].pivot(index="sbc", columns="model", values="srmse").round(4).to_csv(results / "forecast_holdout_srmse.csv")
    pd.DataFrame(summ).T.to_csv(results / "simulation_summary.csv")
    pol = prep["opt"].join(prep["heur"][["s", "S"]].rename(columns={"s": "s_heuristic", "S": "S_heuristic"}))
    pol = pol.join(seg[["abc", "xyz", "sbc"]]).join(prep["fc"][["model", "mu", "sigma"]])
    pol.round(3).to_csv(results / "policy_parameters.csv")
    if len(prep["nv"]):
        prep["nv"].round(3).to_csv(results / "newsvendor_seasonal.csv")

    cost_red = 1 - best["total_cost"] / base["total_cost"]
    inv_red = 1 - best["avg_inventory_value"] / base["avg_inventory_value"]
    metrics = {
        "label": "SYNTHETIC",
        "data_source": ds.source,
        "m5_probe": msg,
        "scale": {
            "skus": int(len(ds.skus)),
            "warehouses": len(cfg["warehouses"]),
            "items": int(len(seg)),
            "history_weeks": prep["H"],
            "test_weeks": prep["T"],
            "seasonal_items": int(seg["seasonal"].sum()),
            "seasonal_items_generated_as_seasonal": int(
                seg.loc[seg["seasonal"], "sku_id"].map(ds.skus.set_index("sku_id")["pattern"]).eq("seasonal").sum()
            ),
            "generated_seasonal_items": int(ds.items["sku_id"].map(ds.skus.set_index("sku_id")["pattern"]).eq("seasonal").sum()),
        },
        "segmentation": {
            "abc_items": seg["abc"].value_counts().sort_index().to_dict(),
            "xyz_items": seg["xyz"].value_counts().sort_index().to_dict(),
            "sbc_items": seg["sbc"].value_counts().sort_index().to_dict(),
        },
        "forecast_model_by_sbc": prep["choice"],
        "forecast_holdout_smae": {r.sbc + "|" + r.model: round(r.smae, 4) for r in prep["fsum"].itertuples()},
        "forecast_holdout_srmse": {r.sbc + "|" + r.model: round(r.srmse, 4) for r in prep["fsum"].itertuples()},
        "quantile_backtest_test_year": qb,
        "simulation": summ,
        "headline": {
            "total_cost_reduction_pct": round(100 * cost_red, 2),
            "inventory_value_reduction_pct": round(100 * inv_red, 2),
            "fill_rate_heuristic": base["fill_rate"],
            "fill_rate_optimised": best["fill_rate"],
            "fill_rate_not_worse": bool(best["fill_rate"] >= base["fill_rate"]),
            "milp_value_vs_greedy_cost_reduction_pct": round(
                100 * (1 - best["total_cost"] / summ["optimised_greedy"]["total_cost"]), 2
            ),
        },
        "milp": milp_diag,
        "budget_per_week": round(inputs["optimised_milp"].budget, 2),
        "capacity_m3": {w["name"]: round(float(c), 2) for w, c in zip(cfg["warehouses"], inputs["optimised_milp"].capacity)},
        "runtime_seconds": {"prepare": round(t_prep, 1), "total": round(time.perf_counter() - t0, 1)},
    }
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    if figures:
        from .report import make_figures

        make_figures(results, prep, runs, inputs, summ, sens, trace_items)
    return metrics
