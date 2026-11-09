"""End-to-end capstone run: world -> history -> Bayesian demand -> calibration -> validation ->
strategic scenarios -> simulation optimisation -> results/.

Usage: python scripts/run_all.py            (full run, writes results/)
       TWIN_QUICK=1 python scripts/run_all.py  (smoke run with tiny budgets, writes to data/quick/)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin30 import calibrate, demand, kpis, report, scenarios, world  # noqa: E402
from twin30.config import (  # noqa: E402
    CATEGORIES,
    DAYS_PER_YEAR,
    HISTORICAL_ASSIGNMENT,
    HISTORY_DAYS,
    SEED,
    SUPPLIERS,
    Decisions,
)
from twin30.twin import Scenario, StartState  # noqa: E402

QUICK = os.environ.get("TWIN_QUICK") == "1"
B = {
    "fit_rows": 1500 if QUICK else 8000,
    "draws": 60 if QUICK else 600,
    "tune": 60 if QUICK else 600,
    "val_reps": 12 if QUICK else 500,
    "scen_reps": 12 if QUICK else 1000,
    "screen_reps": 4 if QUICK else 60,
    "final_reps": 12 if QUICK else 500,
    "top_k": 2 if QUICK else 5,
}
OUT = ROOT / ("data/quick" if QUICK else "results")
FIG = OUT / "figures"
GEN = ROOT / "data" / "generated"
SEEDS = {"pymc": 11, "validation": 10_000, "plan_calendar": 7, "scenarios": 20_000, "screen": 30_000, "final": 40_000}


def load_tables(d: Path) -> dict[str, pd.DataFrame]:
    return {n: pd.read_csv(d / f"{n}.csv.gz") for n in ("sales", "shipments", "purchase_orders", "dc_inventory", "outages")}


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"budgets": B, "seeds": {"world": SEED, **SEEDS}}

    # 1. SYNTHETIC world and its observable history ---------------------------------------------
    run = world.simulate_world(SEED)
    sizes = world.write_history(run, GEN)
    sample_bytes = world.write_sample(run, ROOT / "data" / "sample")
    tabs = load_tables(GEN)  # from here on only the observable tables are used
    snaps = json.loads((GEN / "snapshots.json").read_text())
    metrics["data"] = {
        "path_used": "synthetic_generator",
        "public_attempted": False,
        "reason": "the capstone needs known ground truth (promo effects, lead times, outages); no public dataset has it",
        "history_days": HISTORY_DAYS,
        "history_start": "2024-11-04",
        "tables": sizes,
        "sample_bytes": sample_bytes,
        "stores": 12,
        "categories": len(CATEGORIES),
        "dcs_existing": 2,
        "suppliers": len(SUPPLIERS),
        "outages_in_history": len(tabs["outages"]),
    }
    log(f"history written {sizes}", t0)

    # 2. Bayesian demand + promo model on year 1, evaluated on year 2 -------------------------------
    Y = DAYS_PER_YEAR
    post1, diag1 = demand.fit(tabs["sales"], 0, Y, draws=B["draws"], tune=B["tune"], max_rows=B["fit_rows"], seed=SEEDS["pymc"])
    log(f"PyMC year-1 fit {diag1}", t0)
    truth = run.truth
    naive = demand.naive_promo_lift(tabs["sales"], 0, Y)
    lift = np.exp(post1.promo)
    dip = np.exp(post1.post)
    promo_rows = []
    for c, cat in enumerate(CATEGORIES):
        lo, hi = demand.hdi(lift[:, c])
        dlo, dhi = demand.hdi(dip[:, c])
        promo_rows.append(
            {
                "category": cat.name,
                "true_lift": truth.promo_lift[c],
                "posterior_mean": float(lift[:, c].mean()),
                "hdi90_low": lo,
                "hdi90_high": hi,
                "truth_in_hdi": bool(lo <= truth.promo_lift[c] <= hi),
                "naive_lift": float(naive[c]),
                "true_post_dip": truth.post_promo_dip[c],
                "post_dip_mean": float(dip[:, c].mean()),
                "post_dip_hdi90_low": dlo,
                "post_dip_hdi90_high": dhi,
                "post_truth_in_hdi": bool(dlo <= truth.post_promo_dip[c] <= dhi),
            }
        )
    promo_df = pd.DataFrame(promo_rows)
    promo_df.to_csv(OUT / "promo_effects.csv", index=False)
    report.promo_effects(promo_df, FIG / "promo_effects.png")
    metrics["demand_model"] = {"year1_fit": diag1, "promo_effects": promo_rows}

    # forecast accuracy on the hold-out year (non-stockout store-category-days)
    fc = forecast_eval(tabs, run, post1)
    metrics["demand_model"]["holdout_forecast"] = fc["metrics"]
    report.forecast(fc["weekly"], FIG / "forecast_holdout.png")
    log(f"forecast {fc['metrics']}", t0)

    # 3. calibration + validation of the twin on year 2 -------------------------------------------
    cal1 = calibrate.calibrate(tabs, 0, Y)
    hist = kpis.compute(kpis.ledger_from_tables(tabs, Y, 2 * Y))
    start = StartState.from_snapshot(snaps[str(Y)])
    promo_y2 = run_calendar(tabs, Y, 2 * Y)
    names = [s.name for s in SUPPLIERS]
    replay = [
        (names.index(o.supplier), o.start_day - Y, o.end_day - Y)
        for o in tabs["outages"].itertuples()
        if o.end_day > Y and o.start_day < 2 * Y
    ]
    scen = Scenario(Decisions(), Y, Y, promo_y2[0], promo_y2[1], "replay", replay)
    ctx_val = scenarios.PlanContext(cal1, post1, start, Y, Y, {}, np.zeros(1))
    val_res = scenarios.run_many_custom(ctx_val, scen, B["val_reps"], SEEDS["validation"])
    rows = []
    for k in kpis.VALIDATION_KPIS:
        v = val_res[k].to_numpy()
        h = hist[k]
        rows.append(
            {
                "kpi": k,
                "historical": h,
                "twin_mean": float(v.mean()),
                "twin_p5": float(np.percentile(v, 5)),
                "twin_p95": float(np.percentile(v, 95)),
                "error_pct": float(100 * (v.mean() / h - 1)),
                "within_5pct": bool(abs(v.mean() / h - 1) <= 0.05),
                "historical_inside_twin_5_95": bool(np.percentile(v, 5) <= h <= np.percentile(v, 95)),
            }
        )
    val = pd.DataFrame(rows)
    val.to_csv(OUT / "validation_kpis.csv", index=False)
    report.validation(val, FIG / "validation_kpis.png")
    metrics["validation"] = {
        "calibration_window_days": [0, Y],
        "holdout_window_days": [Y, 2 * Y],
        "reps": B["val_reps"],
        "outages_replayed": replay,
        "calibration": cal1.summary(),
        "kpis": rows,
        "max_abs_error_pct": float(val.error_pct.abs().max()),
        "all_within_5pct": bool(val.within_5pct.all()),
        "n_historical_inside_twin_5_95": int(val.historical_inside_twin_5_95.sum()),
        "true_fill_rate_holdout_hidden": float(run.sales[Y:].sum() / run.demand[Y:].sum()),
        "world_seed_sensitivity": world_seed_sensitivity(),
        "offshore_lead_mean_days": {
            "calibration_year": float(lead_by_year(tabs, 0, Y)),
            "holdout_year": float(lead_by_year(tabs, Y, 2 * Y)),
        },
        "twin_fill_rate_holdout": float(val_res.fill_rate.mean()),
    }
    log(f"validation max |err| {val.error_pct.abs().max():.2f}%", t0)

    # 4. refit on the full history for planning -------------------------------------------------
    post, diag = demand.fit(tabs["sales"], 0, 2 * Y, draws=B["draws"], tune=B["tune"], max_rows=B["fit_rows"], seed=SEEDS["pymc"] + 1)
    cal = calibrate.calibrate(tabs, 0, 2 * Y)
    metrics["demand_model"]["full_history_fit"] = diag
    metrics["plan_calibration"] = cal.summary()
    log(f"PyMC full-history fit {diag}", t0)
    ctx = scenarios.build_context(tabs["sales"], snaps[str(2 * Y)], cal, post, 2 * Y, SEEDS["plan_calendar"])

    # 5. strategic options under three regimes -------------------------------------------------
    strat = scenarios.strategic_options(ctx)
    milp = strat["milp"]
    report.network(HISTORICAL_ASSIGNMENT, milp["forced_east"]["assignment"], FIG / "network.png")
    regimes = ["normal", "stress"]
    dist: dict[str, dict[str, np.ndarray]] = {}
    scen_rows = []
    scen_metrics: dict = {}
    base_runs = {}
    for name, dec in strat["options"].items():
        dist[name] = {}
        scen_metrics[name] = {"decisions": dec_dict(dec)}
        for reg in regimes:
            df = scenarios.run_scenario(ctx, dec, reg, B["scen_reps"], SEEDS["scenarios"])
            if name == "A_baseline":
                base_runs[reg] = df
            s = scenarios.risk_summary(df, base_runs[reg])
            scen_metrics[name][reg] = s
            dist[name][reg] = df.operating_profit.to_numpy()
            scen_rows.append({"scenario": name, "regime": reg, **s})
        log(f"scenario {name} done", t0)

    # 6. optimiser over the decision grid (mixed regime: 1 in 4 years carries the stress outage) -----
    base_screen = scenarios.run_scenario(ctx, Decisions(), "mixed", B["screen_reps"], SEEDS["screen"])
    opt = scenarios.optimise(
        ctx,
        milp["forced_east"]["assignment"],
        base_screen,
        B["screen_reps"],
        B["final_reps"],
        B["top_k"],
        SEEDS["screen"],
        SEEDS["final"],
    )
    opt["screen"].to_csv(OUT / "optimiser_screen.csv", index=False)
    report.frontier(opt["screen"], opt["best_summary"], FIG / "optimiser_frontier.png")
    best = opt["best_decisions"]
    log(f"optimiser best {dec_dict(best)}", t0)
    dist["E_optimised"] = {}
    scen_metrics["E_optimised"] = {"decisions": dec_dict(best)}
    for reg in regimes:
        df = scenarios.run_scenario(ctx, best, reg, B["scen_reps"], SEEDS["scenarios"])
        s = scenarios.risk_summary(df, base_runs[reg])
        scen_metrics["E_optimised"][reg] = s
        dist["E_optimised"][reg] = df.operating_profit.to_numpy()
        scen_rows.append({"scenario": "E_optimised", "regime": reg, **s})
    pd.DataFrame(scen_rows).to_csv(OUT / "scenario_comparison.csv", index=False)
    report.scenarios(dist, FIG / "scenario_profit.png")

    metrics["network_milp"] = {
        "annual_units_plan": float(ctx.annual_units.sum()),
        "status_quo_assignment_cost": milp["status_quo_assignment_cost"],
        "best_without_east": strip(milp["best_without_east"]),
        "forced_east": strip(milp["forced_east"]),
        "free_choice": strip(milp["free_choice"]),
        "milp_opens_east": bool(milp["free_choice"]["open"][2]),
        "milp_saving_vs_status_quo": milp["status_quo_assignment_cost"] - milp["free_choice"]["objective"],
        "milp_saving_forced_east_vs_status_quo": milp["status_quo_assignment_cost"] - milp["forced_east"]["objective"],
    }
    metrics["scenarios"] = {
        "plan_year_start_day": 2 * Y,
        "reps_per_regime": B["scen_reps"],
        "regimes": {
            "normal": "outages drawn from the calibrated Poisson rate posterior",
            "stress": "normal + forced 28-day offshore outage from plan day 14",
        },
        "results": scen_metrics,
        "promo_weeks": {k: v[0][::7].sum(axis=0).astype(int).tolist() for k, v in ctx.calendars.items()},
    }
    metrics["optimiser"] = {k: v for k, v in opt.items() if k not in ("screen", "best_decisions")} | {
        "best_decisions": dec_dict(best),
        "n_feasible_in_screen": int(opt["screen"].feasible.sum()),
        "objective": "max E[operating profit] s.t. CVaR5%(profit) >= baseline CVaR5% and mean fill rate >= 95%",
    }
    metrics["acceptance"] = acceptance(metrics)
    metrics["runtime_s"] = round(time.time() - t0, 1)

    # 7. artefacts for the Streamlit app ----------------------------------------------------------
    model_dir = OUT / "model"
    model_dir.mkdir(exist_ok=True)
    keep = np.linspace(0, post.n - 1, min(200, post.n)).astype(int)
    np.savez_compressed(model_dir / "posterior.npz", **{k: v[keep] for k, v in post.to_npz().items()})
    (model_dir / "plan_context.json").write_text(
        json.dumps(
            {
                "calibration": cal.to_dict(),
                "snapshot": snaps[str(2 * Y)],
                "day0": 2 * Y,
                "horizon": Y,
                "calendars": {k: [v[0][::7].tolist(), v[1][::7].tolist()] for k, v in ctx.calendars.items()},
                "east_assignment": list(milp["forced_east"]["assignment"]),
                "optimised": dec_dict(best),
            }
        )
    )
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json))
    report.one_pager(json.loads((OUT / "metrics.json").read_text()), FIG, OUT / "executive_one_pager.pdf")
    log(f"done; acceptance {metrics['acceptance']}", t0)


def world_seed_sensitivity(seeds=range(1, 9)) -> dict:
    """Year-2 service of the SYNTHETIC world under other seeds (context for the validation bands)."""
    if QUICK:
        seeds = range(1, 3)
    rows = []
    for sd in seeds:
        w = world.simulate_world(sd)
        rows.append(
            {
                "seed": sd,
                "fill_rate": float(w.sales[DAYS_PER_YEAR:].sum() / w.demand[DAYS_PER_YEAR:].sum()),
                "in_stock_rate": float(1 - w.stockout[DAYS_PER_YEAR:].mean()),
            }
        )
    f = [r["fill_rate"] for r in rows]
    i = [r["in_stock_rate"] for r in rows]
    return {"seeds": rows, "fill_rate_range": [min(f), max(f)], "in_stock_rate_range": [min(i), max(i)]}


def lead_by_year(tabs, a: int, b: int) -> float:
    po = tabs["purchase_orders"].drop_duplicates(["order_day", "supplier", "dc"])
    po = po[(po.order_day >= a) & (po.order_day < b) & (po.supplier == SUPPLIERS[0].name)]
    return float((po.receipt_day - po.dispatch_day).mean())


def run_calendar(tabs, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    s = tabs["sales"]
    s = s[(s.day >= a) & (s.day < b)]
    p = s.pivot_table(index="day", columns="category", values="promo", aggfunc="max")[[c.name for c in CATEGORIES]]
    q = s.pivot_table(index="day", columns="category", values="post_promo", aggfunc="max")[[c.name for c in CATEGORIES]]
    return p.to_numpy().astype(float), q.to_numpy().astype(float)


def forecast_eval(tabs, run, post) -> dict:
    Y = DAYS_PER_YEAR
    promo, postf = run_calendar(tabs, Y, 2 * Y)
    mu = scenarios.posterior_mean_demand(post, Y, Y, promo, postf)  # (T, S, C)
    s = tabs["sales"]
    cat = {c.name: i for i, c in enumerate(CATEGORIES)}
    from twin30.config import STORES

    st = {x.name: i for i, x in enumerate(STORES)}
    cube = np.zeros((2 * Y, 12, len(CATEGORIES)))
    so = np.zeros_like(cube)
    cube[s.day, s.store.map(st), s.category.map(cat)] = s.units_sold
    so[s.day, s.store.map(st), s.category.map(cat)] = s.stockout
    act, ok = cube[Y:], so[Y:] == 0
    snaive = cube[:Y]
    legacy = np.stack([cube[Y + t - 28 : Y + t].mean(axis=0) for t in range(Y)])

    def wape(f):
        return float(np.abs(f - act)[ok].sum() / act[ok].sum())

    # 90% predictive interval coverage from posterior-predictive draws
    rng = np.random.default_rng(5)
    idx = np.linspace(0, post.n - 1, min(100, post.n)).astype(int)
    draws = np.stack([post.sample_demand(i, post.mean(i, Y, Y, promo, postf), rng) for i in idx])
    lo, hi = np.percentile(draws, 5, axis=0), np.percentile(draws, 95, axis=0)
    cover = float(((act >= lo) & (act <= hi))[ok].mean())
    wk_draw = np.where(ok[None], draws, 0).sum(axis=(2, 3)).reshape(len(idx), 52, 7).sum(axis=2)
    weekly = pd.DataFrame(
        {
            "week": np.arange(52),
            "actual": np.where(ok, act, 0).sum(axis=(1, 2)).reshape(52, 7).sum(axis=1),
            "forecast": np.where(ok, mu, 0).sum(axis=(1, 2)).reshape(52, 7).sum(axis=1),
            "p5": np.percentile(wk_draw, 5, axis=0),
            "p95": np.percentile(wk_draw, 95, axis=0),
        }
    )
    wk_cover = float(((weekly.actual >= weekly.p5) & (weekly.actual <= weekly.p95)).mean())
    return {
        "metrics": {
            "wape_bayes": wape(mu),
            "wape_seasonal_naive_364": wape(snaive),
            "wape_legacy_trailing_28d": wape(legacy),
            "pi90_coverage_store_day": cover,
            "pi90_coverage_network_week": wk_cover,
            "n_eval_cells": int(ok.sum()),
        },
        "weekly": weekly,
    }


def dec_dict(d: Decisions) -> dict:
    out = asdict(d)
    out["assignment"] = list(d.assignment)
    return out


def strip(m: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in m.items()}


def acceptance(m: dict) -> dict:
    res = m["scenarios"]["results"]
    n_strat = len([k for k in res if k != "A_baseline"])
    return {
        "twin_baseline_kpis_within_5pct": m["validation"]["all_within_5pct"],
        "max_abs_kpi_error_pct": m["validation"]["max_abs_error_pct"],
        "strategic_scenarios_compared_with_uncertainty": n_strat,
        "at_least_3_scenarios": n_strat >= 3,
    }


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Decisions):
        return dec_dict(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    main()
