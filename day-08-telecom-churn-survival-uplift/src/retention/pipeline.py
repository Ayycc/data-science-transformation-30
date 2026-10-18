"""End-to-end pipeline: SYNTHETIC data -> survival models -> CLV -> uplift -> budgeted policy."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from . import clv as clvm
from . import policy as pol
from . import report
from . import survival as sv
from . import uplift as up
from .data import CAUSES, CONTRACTS, GenConfig, design_matrix, generate


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def gen_config(cfg: dict, n_customers: int | None = None) -> GenConfig:
    d = cfg["data"]
    return GenConfig(
        n_customers=n_customers or d["n_customers"],
        n_months=d["n_months"],
        max_start_month=d["max_start_month"],
        dropout_monthly=d["dropout_monthly"],
        campaign_month=d["campaign_month"],
        campaign_window=d["campaign_window"],
        treat_share=d["treat_share"],
        seed=cfg["seed"],
    )


def economics_kwargs(cfg: dict) -> dict:
    e = cfg["economics"]
    return {
        "margin_rate": e["margin_rate"],
        "monthly_discount_rate": e["monthly_discount_rate"],
        "horizon": e["clv_horizon_months"],
        "contact_cost": e["contact_cost"],
        "discount_share": e["discount_share"],
        "discount_months": e["discount_months"],
    }


def make_data(cfg: dict, n_customers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate(gen_config(cfg, n_customers), economics_kwargs(cfg))


def write_sample(panel: pd.DataFrame, truth: pd.DataFrame, out: Path, n: int = 5000) -> None:
    out.mkdir(parents=True, exist_ok=True)
    p = panel.head(n)
    p.to_csv(out / "panel_sample.csv.gz", index=False)
    truth[truth["customer_id"].isin(p["customer_id"])].round(6).to_csv(out / "campaign_truth_sample.csv.gz", index=False)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "data_source": "SYNTHETIC (seeded generator src/retention/data.py)",
                "rows_panel": len(p),
                "note": "First rows of the full 200k-customer panel; make data regenerates everything.",
            },
            indent=2,
        )
    )


def _r(x: float, k: int = 4) -> float:
    return round(float(x), k)


def run(cfg: dict, results: Path, data_dir: Path | None = None, n_customers: int | None = None, fast: bool = False) -> dict:
    t_start = time.time()
    report.reset_style()
    results.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    scfg, ucfg, econ = cfg["survival"], cfg["uplift"], cfg["economics"]
    if fast:  # used by tests: tiny forests
        scfg = {**scfg, "rsf_trees": 10, "rsf_train_n": 2000, "rsf_test_n": 2000, "ph_test_n": 5000}
        ucfg = {**ucfg, "uplift_rf_trees": 5, "uplift_rf_train_n": 3000}
    timings: dict[str, float] = {}

    # ---------------------------------------------------------------- data
    t0 = time.time()
    panel, truth = make_data(cfg, n_customers)
    if data_dir is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
        panel.to_csv(data_dir / "panel.csv.gz", index=False)
        truth.to_csv(data_dir / "campaign_truth.csv.gz", index=False)
    timings["data"] = time.time() - t0
    m: dict = {
        "data_source": "SYNTHETIC (seeded generator; no public download attempted - the prompt specifies a synthetic panel)",
        "scale": {
            "customers": int(len(panel)),
            "calendar_months": cfg["data"]["n_months"],
            "churn_events": int(panel["event"].sum()),
            "censored_share": _r(1 - panel["event"].mean()),
            "campaign_customers": int(panel["in_campaign"].sum()),
            "campaign_treated": int(panel["treatment"].sum()),
        },
    }

    # ---------------------------------------------------------------- survival
    t0 = time.time()
    sf = sv.survival_frame(panel)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(sf))
    n_test = int(scfg["test_share"] * len(sf))
    test, train = sf.iloc[perm[:n_test]], sf.iloc[perm[n_test:]]

    km_contract, p_contract = sv.kaplan_meier_by(sf, "contract", sv.CONTRACT_LABELS)
    km_autopay, p_autopay = sv.kaplan_meier_by(sf, "autopay", {0: "manual pay", 1: "autopay"})
    km_fiber, p_fiber = sv.kaplan_meier_by(sf, "fiber", {0: "DSL/copper", 1: "fiber"})
    km_tab = pd.concat([sv.km_table(c, [6, 12, 24, 36]) for c in (km_contract, km_autopay, km_fiber)])
    km_tab.to_csv(results / "km_by_segment.csv", index=False)
    report.km_panels(
        [("contract", km_contract, p_contract), ("payment", km_autopay, p_autopay), ("network", km_fiber, p_fiber)],
        results / "km_by_segment.png",
    )

    sub = train.iloc[: scfg["ph_test_n"]]
    cox_sub = sv.fit_cox(sub, stratified=False)
    ph_unstrat = sv.ph_test(cox_sub, sub, stratified=False)
    cox_sub_s = sv.fit_cox(sub, stratified=True)
    ph_strat = sv.ph_test(cox_sub_s, sub, stratified=True)
    ph = ph_unstrat.merge(ph_strat, on="covariate", how="left", suffixes=("_unstratified", "_stratified_by_contract"))
    ph.to_csv(results / "ph_test_schoenfeld.csv", index=False)
    report.schoenfeld_plot(
        sv.scaled_schoenfeld(cox_sub, sub, sv.CONTRACT_DUMMIES), sv.CONTRACT_DUMMIES, results / "schoenfeld_contract.png"
    )

    cox = sv.fit_cox(train, stratified=False)
    cox_s = sv.fit_cox(train, stratified=True)
    hr = cox_s.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].rename(
        columns={"exp(coef)": "hazard_ratio"}
    )
    hr.round(5).to_csv(results / "cox_stratified_hazard_ratios.csv")
    rsf = sv.fit_rsf(train.iloc[: scfg["rsf_train_n"]], scfg["rsf_trees"], scfg["rsf_min_leaf"], seed)
    times = np.array(scfg["eval_times"])
    ev_test = test.iloc[: scfg["rsf_test_n"]]
    surv_metrics = {
        "cox_ph": sv.evaluate(train, ev_test, sv.cox_survival_matrix(cox, ev_test, times), times, 12),
        "cox_stratified": sv.evaluate(train, ev_test, sv.cox_survival_matrix(cox_s, ev_test, times), times, 12),
        "random_survival_forest": sv.evaluate(train, ev_test, sv.rsf_survival_matrix(rsf, ev_test, times), times, 12),
    }
    report.td_auc_plot(surv_metrics, results / "time_dependent_auc.png")
    pd.DataFrame(
        [
            {"model": k, **{kk: vv for kk, vv in v.items() if kk != "td_auc"}, **{f"auc_{t}m": a for t, a in v["td_auc"].items()}}
            for k, v in surv_metrics.items()
        ]
    ).round(4).to_csv(results / "survival_metrics.csv", index=False)

    aj = sv.aalen_johansen(sf, cfg["data"]["n_months"])
    aj.round(5).to_csv(results / "competing_risks_cif.csv", index=False)
    report.cif_plot(aj, CAUSES, results / "competing_risks_cif.png")
    cif_at = {
        f"{c}@{t}m": _r(aj.loc[aj["month"] == t, f"cif_{c}"].iloc[0])
        for c in CAUSES
        for t in (12, 24, 36)
        if (aj["month"] == t).any()
    }

    def _p(df: pd.DataFrame, cov: str, col: str) -> float:
        v = df.loc[df["covariate"] == cov, col]
        return float(v.iloc[0]) if len(v) else float("nan")

    non_contract = [c for c in ph["covariate"] if not c.startswith("contract")]
    m["survival"] = {
        "train_customers": len(train),
        "test_customers": len(test),
        "eval_test_customers": len(ev_test),
        "rsf_train_customers": int(min(scfg["rsf_train_n"], len(train))),
        "logrank_p": {"contract": p_contract, "autopay": p_autopay, "fiber": p_fiber},
        "km_median_tenure_months": {
            r["segment"]: (None if np.isinf(r["median"]) else float(r["median"])) for _, r in km_tab.iterrows()
        },
        "ph_test_subsample": int(len(sub)),
        "ph_p_unstratified": {c: _p(ph_unstrat, c, "p_value") for c in ph_unstrat["covariate"]},
        "ph_p_stratified": {c: _p(ph_strat, c, "p_value") for c in ph_strat["covariate"]},
        "ph_min_p_after_stratification": float(ph_strat["p_value"].min()),
        "ph_violations_before": [c for c in ph_unstrat["covariate"] if _p(ph_unstrat, c, "p_value") < 0.01],
        "ph_violations_after": [c for c in ph_strat["covariate"] if _p(ph_strat, c, "p_value") < 0.01],
        "ph_min_p_non_contract_unstratified": float(min(_p(ph_unstrat, c, "p_value") for c in non_contract)),
        "metrics": surv_metrics,
        "competing_risks_cif": cif_at,
    }
    timings["survival"] = time.time() - t0

    # ---------------------------------------------------------------- CLV from survival curves (decision-time model)
    t0 = time.time()
    cm = cfg["data"]["campaign_month"]
    sf_pre = sv.survival_frame(panel, calendar_cutoff=cm)
    cox_pre = sv.fit_cox(sf_pre, stratified=True)
    camp = panel[panel["in_campaign"] == 1].merge(truth, on="customer_id", how="left").reset_index(drop=True)
    Xc = design_matrix(camp)
    Xc["contract"] = camp["contract"].to_numpy()
    tac = camp["tenure_at_campaign"].to_numpy().astype(int)
    H = econ["clv_horizon_months"]
    S = clvm.survival_grid(cox_pre, Xc, t_max_fit=int(sf_pre["duration"].max()), t_needed=int(tac.max()) + H)
    cond = clvm.conditional_survival(S, tac, H)
    ch = camp["monthly_charges"].to_numpy()
    camp["p_churn6_hat"] = 1 - cond[:, cfg["data"]["campaign_window"] - 1]
    camp["clv_hat"] = clvm.clv(cond, ch, econ["margin_rate"], econ["monthly_discount_rate"])
    camp["value_mult"] = clvm.retention_value_multiplier(
        cond, ch, econ["margin_rate"], econ["monthly_discount_rate"], cfg["data"]["campaign_window"]
    )
    camp["cost_hat"] = clvm.offer_cost(
        cond, ch, econ["contact_cost"], econ["discount_share"], econ["discount_months"], econ["monthly_discount_rate"]
    )
    ctrl = camp["treatment"] == 0
    clv_by_contract = camp.groupby("contract")[["clv_hat", "clv0_true"]].mean().round(2)
    clv_by_contract.index = [CONTRACTS[i] for i in clv_by_contract.index]
    clv_by_contract.to_csv(results / "clv_by_contract.csv")
    report.clv_plot(
        camp["clv_hat"].to_numpy(),
        camp["clv0_true"].to_numpy(),
        camp["contract"].to_numpy(),
        CONTRACTS,
        results / "clv_model_vs_true.png",
    )
    m["clv"] = {
        "decision_model": "Cox PH stratified by contract, fit on history censored at calendar month 30 (no campaign outcomes)",
        "horizon_months": H,
        "pearson_r_vs_true": _r(pearsonr(camp["clv_hat"], camp["clv0_true"])[0]),
        "mean_abs_pct_error": _r((np.abs(camp["clv_hat"] - camp["clv0_true"]) / camp["clv0_true"]).mean()),
        "total_clv_hat": _r(camp["clv_hat"].sum(), 0),
        "total_clv_true": _r(camp["clv0_true"].sum(), 0),
        "mean_clv_hat": _r(camp["clv_hat"].mean(), 2),
        "mean_clv_true": _r(camp["clv0_true"].mean(), 2),
        "by_contract": clv_by_contract.to_dict(orient="index"),
        "churn6_auc_control_arm": _r(roc_auc_score(camp.loc[ctrl, "churn_6m"], camp.loc[ctrl, "p_churn6_hat"])),
        "churn6_mean_pred_control": _r(camp.loc[ctrl, "p_churn6_hat"].mean()),
        "churn6_observed_control": _r(camp.loc[ctrl, "churn_6m"].mean()),
    }
    timings["clv"] = time.time() - t0

    # ---------------------------------------------------------------- uplift
    t0 = time.time()
    uf = up.uplift_frame(camp, design_matrix(camp))
    for c in ("tau_true", "dv_true", "cost_true", "net_true", "p_churn6_hat", "clv_hat", "value_mult", "cost_hat"):
        uf[c] = camp[c].to_numpy()
    perm = np.random.default_rng(seed + 1).permutation(len(uf))
    n_hold = int(ucfg["holdout_share"] * len(uf))
    hold = uf.iloc[perm[:n_hold]].reset_index(drop=True)
    trn = uf.iloc[perm[n_hold:]].reset_index(drop=True)  # development half: model fitting + selection
    learners = ["t_learner", "x_learner", "uplift_rf"]
    e = float(uf["treatment"].mean())
    budget = float(econ["budget"])

    # Model selection without touching the holdout: 3-fold CV on the development half, scored by the
    # doubly robust (AIPW) value of the budgeted policy each model would choose (the business objective).
    mu0_t, mu1_t = up.crossfit_outcome_models(trn, seed)
    dr_dev = (
        up.aipw_scores(trn["retained"].to_numpy(), trn["treatment"].to_numpy(), mu0_t, mu1_t, e) * trn["value_mult"].to_numpy()
    )
    folds = np.random.default_rng(seed + 2).integers(0, ucfg["cv_folds"], len(trn))
    cv = {k: {"qini": [], "dr_policy_net": []} for k in learners + ["churn_score"]}
    for f_ in range(ucfg["cv_folds"]):
        tr_, va_ = trn[folds != f_].reset_index(drop=True), trn[folds == f_].reset_index(drop=True)
        pf = up.fit_predict_all(tr_, va_, ucfg, seed + 10 + f_)
        pf["churn_score"] = va_["p_churn6_hat"].to_numpy()
        b_f = budget * len(va_) / len(hold)
        ch_f, vm_f, dr_f = va_["cost_hat"].to_numpy(), va_["value_mult"].to_numpy(), dr_dev[folds == f_]
        for k, s_ in pf.items():
            cv[k]["qini"].append(up.qini_coefficient(va_["retained"].to_numpy(), va_["treatment"].to_numpy(), s_))
            mk = pol.greedy_budget(s_, ch_f, b_f) if k == "churn_score" else pol.value_policy(s_ * vm_f - ch_f, ch_f, b_f)
            cv[k]["dr_policy_net"].append(float((dr_f[mk] - ch_f[mk]).sum() * len(hold) / len(va_)))
    cv_mean = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in cv.items()}
    selected = max(learners, key=lambda k: cv_mean[k]["dr_policy_net"])

    ph_ = up.fit_predict_all(trn, hold, ucfg, seed)
    ph_["churn_score"] = hold["p_churn6_hat"].to_numpy()

    y_h, t_h, tau_h = hold["retained"].to_numpy(), hold["treatment"].to_numpy(), hold["tau_true"].to_numpy()
    oracle_gain = up.true_gain_curve(tau_h, tau_h)
    rows, qcurves, gcurves = [], {}, {}
    for k, s in ph_.items():
        f, g = up.true_gain_curve(tau_h, s)
        rows.append(
            {
                "model": k,
                "cv_qini": cv_mean[k]["qini"],
                "cv_dr_policy_net": cv_mean[k]["dr_policy_net"],
                "holdout_qini": up.qini_coefficient(y_h, t_h, s),
                "holdout_true_gain_auc": float(np.trapezoid(g - f * g[-1], f)),
                "spearman_vs_true_uplift": float(spearmanr(s, tau_h)[0]),
                "true_uplift_top20pct": float(tau_h[np.argsort(-s)[: int(0.2 * len(s))]].mean()),
            }
        )
        qcurves[k] = up.qini_curve(y_h, t_h, s)
        gcurves[k] = (f, g)
    qcurves["random"] = (qcurves[selected][0], qcurves[selected][0] * qcurves[selected][1][-1])
    gcurves["oracle (true uplift)"] = oracle_gain
    gcurves["random"] = (oracle_gain[0], oracle_gain[0] * oracle_gain[1][-1])
    oracle_auc = float(np.trapezoid(oracle_gain[1] - oracle_gain[0] * oracle_gain[1][-1], oracle_gain[0]))
    up_tab = pd.DataFrame(rows)
    up_tab["true_gain_vs_oracle"] = up_tab["holdout_true_gain_auc"] / oracle_auc
    up_tab.round(5).to_csv(results / "uplift_metrics.csv", index=False)
    report.qini_plot(
        qcurves,
        results / "qini_curves.png",
        f"Qini curves on holdout, observed outcomes ({report.SYN})",
        "incremental retained customers / N",
    )
    report.qini_plot(
        gcurves,
        results / "uplift_true_gain.png",
        f"Cumulative TRUE uplift on holdout ({report.SYN})",
        "true incremental retained / N",
    )
    dec = pd.qcut(-ph_[selected], 10, labels=False, duplicates="drop")
    dec_rows = []
    for d_ in range(int(dec.max()) + 1):
        mk = dec == d_
        dec_rows.append(
            {
                "decile": d_ + 1,
                "observed_uplift": y_h[mk & (t_h == 1)].mean() - y_h[mk & (t_h == 0)].mean(),
                "true_uplift": tau_h[mk].mean(),
                "n": int(mk.sum()),
            }
        )
    dec_df = pd.DataFrame(dec_rows)
    dec_df.round(5).to_csv(results / "uplift_by_decile.csv", index=False)
    report.uplift_by_decile(dec_df, results / "uplift_by_decile.png")
    m["uplift"] = {
        "train_customers": len(trn),
        "cv_folds": ucfg["cv_folds"],
        "holdout_customers": len(hold),
        "uplift_rf_train_customers": int(min(ucfg["uplift_rf_train_n"], len(trn))),
        "true_ate_holdout": _r(tau_h.mean()),
        "observed_ate_holdout": _r(y_h[t_h == 1].mean() - y_h[t_h == 0].mean()),
        "true_sleeping_dog_share": _r((tau_h < 0).mean()),
        "selected_model_by_cv_dr_policy_value": selected,
        "models": {r["model"]: {k: _r(v, 5) for k, v in r.items() if k != "model"} for r in up_tab.to_dict(orient="records")},
    }
    timings["uplift"] = time.time() - t0

    # ---------------------------------------------------------------- policies
    t0 = time.time()
    mu_models = up.fit_outcome_models(trn, seed)
    Xh = hold[up.UPLIFT_FEATURES].to_numpy()
    mu0_h, mu1_h = (mm.predict_proba(Xh)[:, 1] for mm in mu_models)
    hold["dr_value"] = up.aipw_scores(y_h, t_h, mu0_h, mu1_h, e) * hold["value_mult"].to_numpy()
    cost_hat, vm = hold["cost_hat"].to_numpy(), hold["value_mult"].to_numpy()

    # DR policy learning (stretch): shallow tree on cross-fitted AIPW net-value scores from the development half
    pseudo = dr_dev - trn["cost_hat"].to_numpy()
    tree = pol.fit_policy_tree(trn[up.UPLIFT_FEATURES], pseudo, depth=3, seed=seed)
    tree_net = pol.leaf_values(tree, trn[up.UPLIFT_FEATURES], pseudo, hold[up.UPLIFT_FEATURES])

    def policies(b: float) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
        out = []
        for k in learners:
            nh = ph_[k] * vm - cost_hat
            out.append((f"uplift value ({k})", pol.value_policy(nh, cost_hat, b), nh))
        out.append(("churn score (Cox 6-month risk)", pol.greedy_budget(hold["p_churn6_hat"].to_numpy(), cost_hat, b), None))
        out.append(
            (
                "value at risk (churn risk x CLV)",
                pol.greedy_budget(hold["p_churn6_hat"].to_numpy() * hold["clv_hat"].to_numpy(), cost_hat, b),
                None,
            )
        )
        out.append(("DR policy tree (depth 3)", pol.value_policy(tree_net, cost_hat, b), None))
        out.append(("random", pol.greedy_budget(np.random.default_rng(seed + 7).random(len(hold)), cost_hat, b), None))
        out.append(
            ("oracle (true net value)", pol.value_policy(hold["net_true"].to_numpy(), hold["cost_true"].to_numpy(), b), None)
        )
        return out

    ptab = pd.DataFrame([pol.evaluate_policy(n, mk, hold, nh, cost_hat) for n, mk, nh in policies(budget)])
    ptab = pd.concat(
        [ptab, pd.DataFrame([pol.evaluate_policy("treat everyone (no budget)", np.ones(len(hold), bool), hold, None, cost_hat)])],
        ignore_index=True,
    )
    ptab.round(4).to_csv(results / "policy_comparison.csv", index=False)
    report.policy_bars(ptab, results / "policy_comparison.png")

    bc_rows = []
    for b in [100e3, 200e3, 400e3, 600e3, 800e3, 1.2e6]:
        for n, mk, nh in policies(b):
            if n.startswith("uplift value") and selected not in n:
                continue
            if n in ("random", "DR policy tree (depth 3)"):
                continue
            r = pol.evaluate_policy(n, mk, hold, nh, cost_hat)
            bc_rows.append(
                {
                    "budget": b,
                    "policy": n,
                    "net_incremental_revenue_true": r["net_incremental_revenue_true"],
                    "n_targeted": r["n_targeted"],
                }
            )
    bc = pd.DataFrame(bc_rows)
    bc.round(2).to_csv(results / "budget_curve.csv", index=False)
    report.budget_curve(bc, results / "budget_curve.png")

    pr = {
        r["policy"]: {k: (_r(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "policy"}
        for r in ptab.to_dict(orient="records")
    }
    sel_name = f"uplift value ({selected})"
    churn_name = "churn score (Cox 6-month risk)"
    m["policy"] = {
        "budget": budget,
        "offer": f"{int(econ['discount_share'] * 100)}% off for {econ['discount_months']} months"
        f" + ${econ['contact_cost']:.0f} contact",
        "selected_policy": sel_name,
        "policy_tree_rules": _tree_text(tree, up.UPLIFT_FEATURES),
        "policies": pr,
    }
    timings["policy"] = time.time() - t0

    s_net, c_net = pr[sel_name]["net_incremental_revenue_true"], pr[churn_name]["net_incremental_revenue_true"]
    sm = surv_metrics
    m["headline"] = {
        "uplift_policy_net_true": s_net,
        "churn_policy_net_true": c_net,
        "uplift_minus_churn_net": _r(s_net - c_net, 2),
        "uplift_policy_roi_true": pr[sel_name]["roi_true"],
        "uplift_policy_expected_roi": pr[sel_name].get("expected_roi"),
        "churn_policy_roi_true": pr[churn_name]["roi_true"],
        "uplift_beats_churn_score": bool(s_net > c_net),
        "best_survival_model_uno_c": max(sm, key=lambda k: sm[k]["uno_c"]),
        "cox_stratified_uno_c": _r(sm["cox_stratified"]["uno_c"]),
        "cox_stratified_td_auc_mean": _r(sm["cox_stratified"]["td_auc_mean"]),
        "rsf_uno_c": _r(sm["random_survival_forest"]["uno_c"]),
        "ph_violations_before": m["survival"]["ph_violations_before"],
        "ph_violations_after": m["survival"]["ph_violations_after"],
        "clv_pearson_r": m["clv"]["pearson_r_vs_true"],
    }
    timings["total"] = time.time() - t_start
    m["runtime_seconds"] = {k: round(v, 1) for k, v in timings.items()}
    (results / "metrics.json").write_text(json.dumps(_clean(m), indent=2))
    return m


def _tree_text(tree, names: list[str]) -> str:
    from sklearn.tree import export_text

    return export_text(tree, feature_names=names, decimals=2)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating | float):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
