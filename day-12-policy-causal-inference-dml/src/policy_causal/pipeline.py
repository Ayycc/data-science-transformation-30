"""End-to-end pipeline: simulate -> identify -> estimate -> refute -> DiD -> synthetic control -> memo."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import dag, did, dml, plots, propensity, refute, synth
from .config import CONFOUNDERS, DATA_SAMPLE, FIGURES, MODIFIERS, RESULTS, RunConfig
from .lalonde import load_lalonde, run_lalonde
from .memo import write_memo, write_report
from .simulate import observed, panel_truth, simulate_individuals, simulate_panel, true_effects

OUTCOME_LABEL = {"earnings_post": "earnings", "employed_post": "employment"}


def naive_diff(df: pd.DataFrame, y: str) -> dict[str, float]:
    t = df["treated"] == 1
    a, b = df.loc[t, y].to_numpy(float), df.loc[~t, y].to_numpy(float)
    est = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"estimate": float(est), "se": float(se), "ci_low": float(est - 1.96 * se), "ci_high": float(est + 1.96 * se)}


def ols(df: pd.DataFrame, y: str, covs: list[str]) -> dict[str, float]:
    x = sm.add_constant(df[["treated", *covs]].astype(float))
    r = sm.OLS(df[y].astype(float), x).fit(cov_type="HC1")
    b, se = float(r.params["treated"]), float(r.bse["treated"])
    return {"estimate": b, "se": se, "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se}


def _clean(o: object) -> object:
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


def run(cfg: RunConfig, results_dir: Path = RESULTS, figures_dir: Path = FIGURES, lalonde_source: str = "auto") -> dict[str, object]:
    t_start = time.time()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def lap(name: str, t0: float) -> float:
        timings[name] = round(time.time() - t0, 1)
        return time.time()

    t0 = time.time()
    full = simulate_individuals(cfg.n_people, cfg.seed)
    df = observed(full)
    truth = true_effects(full)
    panel = simulate_panel(cfg.n_counties, cfg.n_quarters, cfg.cohorts, cfg.seed)
    ptruth = panel_truth(panel)
    t0 = lap("simulate", t0)

    # 1. DAG + identification ------------------------------------------------------------------
    dag.draw_dag(figures_dir / "dag.png")
    ident = dag.dowhy_identification(df, "earnings_post")
    adj = dag.adjustment_report(CONFOUNDERS)
    t0 = lap("identify", t0)

    outcomes = ["earnings_post", "employed_post"]
    controls = [c for c in CONFOUNDERS if c not in MODIFIERS]
    rows: list[dict[str, object]] = []

    def add(y: str, method: str, r: dict[str, float], family: str) -> None:
        tv = truth[f"ate_{OUTCOME_LABEL[y]}"]
        rows.append(
            {
                "outcome": y,
                "method": method,
                "family": family,
                "estimate": r["estimate"],
                "ci_low": r.get("ci_low"),
                "ci_high": r.get("ci_high"),
                "true_ate": tv,
                "bias": r["estimate"] - tv,
                "truth_in_ci": bool(r.get("ci_low") is not None and r["ci_low"] <= tv <= r["ci_high"]),
            }
        )

    # 2. naive and regression ------------------------------------------------------------------
    for y in outcomes:
        add(y, "Naive difference in means", naive_diff(df, y), "naive")
        add(y, "OLS + confounders", ols(df, y, CONFOUNDERS), "regression")
        add(y, "OLS + confounders + certificate (bad control)", ols(df, y, [*CONFOUNDERS, "certificate"]), "regression")
    t0 = lap("naive_ols", t0)

    # 3. propensity methods --------------------------------------------------------------------
    ps = propensity.run_propensity(df, CONFOUNDERS, outcomes, cfg.n_boot, cfg.seed)
    for y in outcomes:
        add(y, "IPW (Hajek, logistic PS)", ps["estimates"][y]["ipw_ate"], "propensity")
        add(y, "PS matching 1:1, caliper 0.2 SD", ps["estimates"][y]["match_ate"], "propensity")
    ps["balance"].to_csv(results_dir / "balance.csv", float_format="%.4f")
    plots.ps_overlap(ps["propensity"], df["treated"].to_numpy(), figures_dir / "ps_overlap.png")
    plots.love_plot(ps["balance"], figures_dir / "love_plot.png")
    t0 = lap("propensity", t0)

    # 4. double ML -----------------------------------------------------------------------------
    dml_res: dict[str, dict[str, object]] = {}
    cate_res: dict[str, dict[str, object]] = {}
    for y in outcomes:
        lab = OUTCOME_LABEL[y]
        ow = dml.overlap_weighted_dml(df, y, CONFOUNDERS, cfg.dml_folds, cfg.seed, cfg.quick)
        lin = dml.linear_dml_ate(df, y, MODIFIERS, controls, cfg.dml_folds, cfg.seed, cfg.quick)
        drl = dml.dr_learner_ate(df, y, CONFOUNDERS, cfg.dml_folds, cfg.seed, cfg.quick)
        add(y, "DML partialling-out, constant effect", ow, "dml")
        add(y, "DML LinearDML, CATE linear in X (headline)", lin, "dml")
        add(y, "DML DR-learner (cross-fitted AIPW)", drl, "dml")
        cf, cf_sum, cate = dml.causal_forest(df, y, MODIFIERS, controls, cfg.dml_folds, cfg.forest_trees, cfg.seed, cfg.quick)
        add(y, "Causal forest (DR average of CATE)", cf_sum["ate_dr"], "dml")
        tt = full[f"true_tau_{lab}"].to_numpy()
        sub = dml.subgroup_table(cf, df, MODIFIERS, cate, tt)
        sub.to_csv(results_dir / f"cate_subgroups_{lab}.csv", index=False, float_format="%.4f")
        tg = dml.targeting_gain(cate, tt, 0.3)
        dml_res[y] = {"overlap_weighted": ow, "linear_dml": lin, "dr_learner": drl, "causal_forest": cf_sum["ate_dr"]}
        cate_res[y] = {
            "feature_importance": cf_sum["feature_importance"],
            "targeting_top30": tg,
            "subgroups_truth_in_ci": int(sub["truth_in_ci"].sum()),
            "subgroups_total": len(sub),
            "subgroups": sub.to_dict(orient="records"),
            "cate_rmse_vs_true": float(np.sqrt(np.mean((cate - tt) ** 2))),
        }
        if y == "earnings_post":
            plots.cate_figure(cate, tt, sub, figures_dir / "cate_earnings.png", cfg.seed)
    cov = dml.coverage_study(
        cfg.n_people, [cfg.seed + k for k in range(1, cfg.coverage_reps + 1)], MODIFIERS, controls, cfg.dml_folds, cfg.quick
    )
    cov["table"].to_csv(results_dir / "coverage_study.csv", index=False, float_format="%.4f")
    t0 = lap("dml", t0)

    # 5. refutations ---------------------------------------------------------------------------
    refs: dict[str, object] = {}
    for y in outcomes:
        refs[y] = refute.dowhy_dml_refutations(df, y, MODIFIERS, cfg.dml_folds, cfg.refute_sims, cfg.seed, cfg.quick)
        add(y, "DoWhy + econml LinearDML (minimal backdoor set)", {"estimate": refs[y]["dowhy_estimate"]}, "dml")
    refs["negative_control_outcome"] = refute.negative_control_outcome(df, MODIFIERS, CONFOUNDERS, cfg.dml_folds, cfg.seed, cfg.quick)
    ref_rows = []
    for y in outcomes:
        for name in ("placebo_treatment", "random_common_cause", "data_subset_80pct"):
            r = refs[y][name]
            ref_rows.append({"outcome": y, "test": name, **r})
    nc = refs["negative_control_outcome"]
    ref_rows.append(
        {
            "outcome": "prior_earn_2",
            "test": "negative_control_outcome",
            "original": None,
            "new_effect": nc["estimate"],
            "pass_rule": nc["pass_rule"],
            "passed": nc["passed"],
        }
    )
    pd.DataFrame(ref_rows).to_csv(results_dir / "refutations.csv", index=False, float_format="%.4f")
    t0 = lap("refute", t0)

    # 6. E-values (sensitivity to unobserved confounding) --------------------------------------
    lin_e = dml_res["employed_post"]["linear_dml"]
    p0 = ps["estimates"]["employed_post"]["ipw_mean_y0"]["estimate"]
    ev_emp = refute.evalue_from_risks(p0 + lin_e["estimate"], p0, lin_e["ci_low"])
    ev_emp["baseline_p0_ipw"] = p0
    lin_y = dml_res["earnings_post"]["linear_dml"]
    ev_earn = refute.evalue_continuous(lin_y["estimate"], lin_y["ci_low"], float(df["earnings_post"].std()))
    t0 = lap("evalues", t0)

    est = pd.DataFrame(rows)
    est.to_csv(results_dir / "estimates.csv", index=False, float_format="%.4f")
    plots.estimator_forest(est, {y: truth[f"ate_{OUTCOME_LABEL[y]}"] for y in outcomes}, figures_dir / "estimator_comparison.png")

    # 7. staggered DiD --------------------------------------------------------------------------
    cs = did.callaway_santanna(panel, control="never", n_boot=cfg.n_boot_did, seed=cfg.seed)
    cs_ny = did.callaway_santanna(panel, control="notyet", n_boot=cfg.n_boot_did, seed=cfg.seed)
    tw = did.twfe(panel)
    tw_es = did.twfe_event_study(panel)
    ev = cs["event_study"].copy()
    ev["true_effect"] = ev["event_time"].map(lambda e: ptruth["event_time"].get(int(e), 0.0))
    ev.to_csv(results_dir / "event_study.csv", index=False, float_format="%.4f")
    pd.DataFrame([{"cohort": g, "quarter": t, "att": v} for (g, t), v in cs["att_gt"].items()]).to_csv(
        results_dir / "att_gt.csv", index=False, float_format="%.4f"
    )
    plots.event_study(cs["event_study"], ptruth["event_time"], tw_es, figures_dir / "event_study.png")
    post_ev = ev[ev["event_time"] >= 0]
    ev_cover = float(np.mean((post_ev["ci_low"] <= post_ev["true_effect"]) & (post_ev["true_effect"] <= post_ev["ci_high"])))
    t0 = lap("did", t0)

    # 8. synthetic control for a single pilot county --------------------------------------------
    first = panel.groupby("county_id")["first_treated"].first()
    early = first[first == min(cfg.cohorts)].index.to_numpy()
    pilot = int(early[0])
    sc = synth.synthetic_control(panel, pilot)
    sc["series"].to_csv(results_dir / "synthetic_control.csv", index=False, float_format="%.4f")
    plots.synth_figure(sc["series"], sc["adoption_quarter"], figures_dir / "synthetic_control.png")
    t0 = lap("synth", t0)

    # 9. optional public LaLonde replication ----------------------------------------------------
    frames, lpath = load_lalonde(lalonde_source)
    lal = run_lalonde(frames, cfg.seed, n_boot=cfg.n_boot) if frames is not None else None
    t0 = lap("lalonde", t0)

    # ---- metrics --------------------------------------------------------------------------------
    head = est[est["method"].str.contains("headline")].set_index("outcome")
    naive = est[est["family"] == "naive"].set_index("outcome")
    metrics: dict[str, object] = {
        "data": {
            "individual_path": "synthetic (seeded generator; by design, truth must be known)",
            "panel_path": "synthetic (seeded generator)",
            "lalonde_path": lpath,
            "seed": cfg.seed,
            "n_people": cfg.n_people,
            "n_counties": cfg.n_counties,
            "n_quarters": cfg.n_quarters,
            "cohorts": {str(k): v for k, v in cfg.cohorts.items()},
            "never_treated_counties": cfg.n_counties - sum(cfg.cohorts.values()),
            "share_enrolled": float(df["treated"].mean()),
            "quick_mode": cfg.quick,
        },
        "truth": {**truth, "panel_att_overall": ptruth["att_overall"]},
        "identification": {
            "dowhy_backdoor_set": ident["backdoor_variables"],
            "dowhy_instruments": ident["instrumental_variables"],
            "adjustment_set_used": CONFOUNDERS,
            "checks": adj,
        },
        "acceptance": {
            "headline_estimator": "DML LinearDML, CATE linear in X",
            "earnings_ate": head.loc["earnings_post", "estimate"],
            "earnings_ci": [head.loc["earnings_post", "ci_low"], head.loc["earnings_post", "ci_high"]],
            "earnings_truth_in_ci": bool(head.loc["earnings_post", "truth_in_ci"]),
            "employment_ate": head.loc["employed_post", "estimate"],
            "employment_ci": [head.loc["employed_post", "ci_low"], head.loc["employed_post", "ci_high"]],
            "employment_truth_in_ci": bool(head.loc["employed_post", "truth_in_ci"]),
            "naive_bias_earnings": naive.loc["earnings_post", "bias"],
            "naive_bias_employment": naive.loc["employed_post", "bias"],
            "naive_bias_earnings_pct_of_truth": 100 * naive.loc["earnings_post", "bias"] / truth["ate_earnings"],
            "naive_bias_employment_pct_of_truth": 100 * naive.loc["employed_post", "bias"] / truth["ate_employment"],
            "refutation_tests_reported": len(ref_rows),
            "refutation_tests_passed": int(sum(bool(r["passed"]) for r in ref_rows)),
        },
        "estimates": est.to_dict(orient="records"),
        "propensity_diagnostics": ps["diagnostics"],
        "dml": dml_res,
        "coverage_study": cov["summary"],
        "cate": cate_res,
        "refutations": refs,
        "evalues": {"employment": ev_emp, "earnings": ev_earn},
        "did": {
            "true_att_overall": ptruth["att_overall"],
            "cs_never_treated": {
                k: cs[k]
                for k in (
                    "att_overall",
                    "att_overall_se",
                    "att_overall_ci",
                    "att_by_cohort",
                    "pretrend_wald",
                    "pretrend_df",
                    "pretrend_p",
                    "pretrend_max_abs",
                    "n_boot_effective",
                )
            },
            "cs_not_yet_treated": {k: cs_ny[k] for k in ("att_overall", "att_overall_se", "att_overall_ci", "pretrend_p")},
            "twfe_static": tw,
            "twfe_bias": tw["estimate"] - ptruth["att_overall"],
            "cs_bias": cs["att_overall"] - ptruth["att_overall"],
            "cs_truth_in_ci": bool(cs["att_overall_ci"][0] <= ptruth["att_overall"] <= cs["att_overall_ci"][1]),
            "twfe_truth_in_ci": bool(tw["ci_low"] <= ptruth["att_overall"] <= tw["ci_high"]),
            "event_time_ci_coverage_post": ev_cover,
            "true_att_by_cohort": {
                str(g): float(panel[(panel["first_treated"] == g) & (panel["treated_now"] == 1)]["true_effect"].mean()) for g in cfg.cohorts
            },
        },
        "synthetic_control": {k: v for k, v in sc.items() if k != "series"},
        "lalonde": lal,
        "runtime": {"seconds_total": None, "stages": timings, "python": platform.python_version(), "cpus": os.cpu_count()},
    }
    metrics["runtime"]["seconds_total"] = round(time.time() - t_start, 1)
    metrics = _clean(metrics)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_report(metrics, est, results_dir / "report.md")
    write_memo(metrics, results_dir / "policy_memo.md")
    return metrics


def write_samples(cfg: RunConfig, folder: Path = DATA_SAMPLE, n_rows: int = 3000) -> None:
    """Small committed samples (SYNTHETIC) so the data can be inspected without running anything."""
    folder.mkdir(parents=True, exist_ok=True)
    full = simulate_individuals(cfg.n_people, cfg.seed)
    full.head(n_rows).to_csv(folder / "individuals_sample.csv", index=False, float_format="%.4f")
    simulate_panel(cfg.n_counties, cfg.n_quarters, cfg.cohorts, cfg.seed).to_csv(
        folder / "county_panel.csv", index=False, float_format="%.4f"
    )
