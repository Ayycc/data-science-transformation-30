"""End-to-end: SYNTHETIC data -> prior predictive -> NUTS fit -> diagnostics -> recovery -> ROAS -> budget optimiser -> results/."""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402
import pymc_marketing  # noqa: E402

from bayes_mmm import analysis, plots  # noqa: E402
from bayes_mmm.config import CHANNELS, FIGURES, HDI_PROB, L_MAX, RESULTS, SamplerConfig  # noqa: E402
from bayes_mmm.data import get_panel, save  # noqa: E402
from bayes_mmm.model import build_model, fit, prepare_inputs, sample_prior  # noqa: E402
from bayes_mmm.optimiser import ResponseModel, linear_allocation, optimise, summarise_lift  # noqa: E402

BOUNDS = (0.5, 1.5)  # each channel may move between 50% and 150% of its current spend
RISK_AVERSION = 1.0  # risk-adjusted objective: mean - 1 sd of posterior contribution


def r(x: float, n: int = 4) -> float:
    return float(round(float(x), n))


def main() -> None:
    logging.getLogger("pymc").setLevel(logging.WARNING)
    t_start = time.time()
    cfg = SamplerConfig.from_env()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    panel, path_used = get_panel()
    save(panel)
    inp = prepare_inputs(panel)
    model = build_model(inp)
    print(f"[data] {path_used}: {panel.n_weeks} weeks x {len(panel.regions)} regions x {len(CHANNELS)} channels (SYNTHETIC)")

    t0 = time.time()
    prior = sample_prior(model, cfg)
    y_prior = prior.prior_predictive["y"].stack(s=("chain", "draw")).transpose("s", ...).to_numpy()
    prior_check = analysis.predictive_check(y_prior, panel, inp)
    t_prior = time.time() - t0
    print(f"[prior] 94% prior predictive covers {prior_check['coverage_94_region_week']:.3f} of region-weeks")

    t0 = time.time()
    idata = fit(model, cfg)
    t_fit = time.time() - t0
    print(f"[fit] {cfg.chains} chains x {cfg.draws} draws (+{cfg.tune} tune) in {t_fit:.0f} s")

    conv = analysis.convergence(idata)
    conv["table"].to_csv(RESULTS / "convergence.csv", index=False)
    print(f"[diag] max R-hat {conv['max_rhat']:.4f}, min ESS bulk {conv['min_ess_bulk']:.0f}, divergences {conv['divergences']}")

    y_post = analysis.flat(idata, "y", group="posterior_predictive")
    post_check = analysis.predictive_check(y_post, panel, inp)

    rec = analysis.recovery(idata, panel, inp)
    rec.to_csv(RESULTS / "parameter_recovery.csv", index=False)
    roas = analysis.roas_table(idata, panel)
    roas.to_csv(RESULTS / "roas.csv", index=False)
    print(roas[["channel", "true_roas", "roas_mean", "roas_hdi_low", "roas_hdi_high", "last_click_roas"]].round(3).to_string(index=False))

    # ---- budget optimisation -------------------------------------------------------------------------
    pop_total = float(panel.population.sum())
    last52 = panel.spend[-52:].sum(axis=(0, 1))  # $k per channel, last 52 weeks
    s0 = last52 / 52 / pop_total  # weekly per-capita spend
    lower, upper = s0 * BOUNDS[0], s0 * BOUNDS[1]
    rm = ResponseModel(
        beta=analysis.flat(idata, "beta"), lam=analysis.flat(idata, "lam"), population=panel.population, y_scale=inp.y_scale, channel_scale=inp.channel_scale
    )
    truth_vals = analysis.true_values(panel, inp)
    rm_true = ResponseModel(
        beta=truth_vals["beta"][None], lam=truth_vals["lam"][None], population=panel.population, y_scale=inp.y_scale, channel_scale=inp.channel_scale
    )
    s_opt = optimise(rm, s0, lower, upper, risk_aversion=0.0)
    s_risk = optimise(rm, s0, lower, upper, risk_aversion=RISK_AVERSION)
    s_lc = linear_allocation(np.array(panel.truth["last_click"]["roas"]), s0, lower, upper)
    s_true_opt = optimise(rm_true, s0, lower, upper, risk_aversion=0.0)

    allocations = {"current": s0, "expected_optimal": s_opt, "risk_adjusted": s_risk, "last_click_guided": s_lc, "oracle_true_optimal": s_true_opt}
    alloc_df = pd.DataFrame({"channel": list(CHANNELS)})
    for k, s in allocations.items():
        alloc_df[k] = s * pop_total * 52  # annual $k
    alloc_df.to_csv(RESULTS / "budget_allocation.csv", index=False)

    base_true = float(rm_true.draws(s0)[0] * 52)
    budget = {}
    lift_draws = {}
    for k in ("expected_optimal", "risk_adjusted", "last_click_guided", "oracle_true_optimal"):
        s = allocations[k]
        summ = summarise_lift(rm, s, s0)
        true_lift = float(rm_true.draws(s)[0] * 52 - base_true)
        summ.update(
            {
                "true_lift_k": true_lift,
                "true_lift_pct_of_media_contribution": true_lift / base_true,
                "true_lift_pct_of_total_sales": true_lift / (panel.sales[-52:].sum()),
            }
        )
        budget[k] = {kk: r(v) for kk, v in summ.items()}
        if k != "oracle_true_optimal":
            lift_draws[k] = (rm.draws(s) - rm.draws(s0)) * 52
    cur = rm.draws(s0) * 52
    clo, chi = analysis.hdi(cur)

    # ---- figures ---------------------------------------------------------------------------------------
    dates = panel.dates
    obs_nat = panel.sales.sum(axis=1)
    prior_nat = analysis.to_sales_k(y_prior, inp).sum(axis=2)
    post_nat = analysis.to_sales_k(y_post, inp).sum(axis=2)
    plots.predictive_bands(dates, obs_nat, prior_nat, post_nat, FIGURES / "predictive_checks.png")
    plots.roas_forest(roas, analysis.flat(idata, "roas"), FIGURES / "roas_posterior.png")
    plots.recovery_plot(rec, FIGURES / "parameter_recovery.png")
    contrib_mean = analysis.flat(idata, "channel_contribution").mean(axis=0)
    plots.contributions_plot(dates, post_nat.mean(axis=0) - contrib_mean.sum(axis=1), contrib_mean, FIGURES / "contributions.png")
    plots.response_curves(rm, rm_true, s0, s_opt, s_risk, FIGURES / "response_curves.png")
    plots.budget_plot(alloc_df, lift_draws, FIGURES / "budget_optimisation.png")
    az.plot_rank(idata, var_names=["alpha", "lam"])
    import matplotlib.pyplot as plt

    plt.gcf().savefig(FIGURES / "rank_alpha_lam.png", dpi=80, bbox_inches="tight")
    plt.close("all")

    # ---- metrics ----------------------------------------------------------------------------------------
    roas_in = int(roas["true_in_hdi"].sum())
    groups = rec.groupby("group")["in_hdi"].agg(["sum", "count"])
    exp_opt = budget["expected_optimal"]
    metrics = {
        "data": {
            "path_used": path_used,
            "label": "SYNTHETIC",
            "seed": panel.truth["seed"],
            "n_weeks": panel.n_weeks,
            "n_regions": len(panel.regions),
            "channels": list(CHANNELS),
            "n_observations": int(panel.sales.size),
            "total_sales_k": r(panel.sales.sum(), 1),
            "total_spend_k": r(panel.spend.sum(), 1),
            "spend_by_channel_k": {c: r(v, 1) for c, v in zip(CHANNELS, panel.spend.sum(axis=(0, 1)), strict=True)},
        },
        "sampler": {
            "draws": cfg.draws,
            "tune": cfg.tune,
            "chains": cfg.chains,
            "target_accept": cfg.target_accept,
            "init": cfg.init,
            "adstock_l_max": L_MAX,
            "hdi_prob": HDI_PROB,
            "fit_seconds": r(t_fit, 1),
            "prior_predictive_seconds": r(t_prior, 1),
        },
        "convergence": {k: (r(v) if isinstance(v, float) else v) for k, v in conv.items() if k != "table"},
        "prior_predictive": {k: r(v) for k, v in prior_check.items()},
        "posterior_predictive": {k: r(v) for k, v in post_check.items()},
        "recovery": {
            "roas_true_in_94_hdi": f"{roas_in}/{len(CHANNELS)}",
            "by_group_in_94_hdi": {g: f"{int(row['sum'])}/{int(row['count'])}" for g, row in groups.iterrows()},
            "all_parameters_in_94_hdi": f"{int(rec['in_hdi'].sum())}/{len(rec)}",
        },
        "roas": {
            row["channel"]: {
                "true": r(row["true_roas"]),
                "mean": r(row["roas_mean"]),
                "hdi_low": r(row["roas_hdi_low"]),
                "hdi_high": r(row["roas_hdi_high"]),
                "true_in_hdi": bool(row["true_in_hdi"]),
                "last_click": r(row["last_click_roas"]),
                "contribution_share_of_sales": r(row["contribution_share_of_sales"]),
            }
            for _, row in roas.iterrows()
        },
        "budget": {
            "planning_horizon_weeks": 52,
            "annual_budget_k": r(s0.sum() * pop_total * 52, 1),
            "channel_bounds_vs_current": list(BOUNDS),
            "risk_aversion_sd": RISK_AVERSION,
            "current_media_contribution_mean_k": r(cur.mean(), 1),
            "current_media_contribution_hdi_k": [r(clo, 1), r(chi, 1)],
            "current_true_media_contribution_k": r(base_true, 1),
            "allocations_annual_k": {k: {c: r(v * pop_total * 52, 1) for c, v in zip(CHANNELS, s, strict=True)} for k, s in allocations.items()},
            **budget,
        },
        "acceptance": {
            "all_rhat_below_1_01": conv["max_rhat"] < 1.01,
            "true_roas_in_94_hdi_for_most_channels": roas_in > len(CHANNELS) / 2,
            "optimised_budget_lift_with_uncertainty": exp_opt["lift_hdi_low_k"] > 0,
        },
        "versions": {
            "python": platform.python_version(),
            "pymc": pm.__version__,
            "pymc_marketing": pymc_marketing.__version__,
            "arviz": az.__version__,
            "numpy": np.__version__,
        },
        "runtime_seconds": r(time.time() - t_start, 1),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_report(metrics, roas, alloc_df, conv["table"])
    print(json.dumps(metrics["acceptance"]), f"runtime {metrics['runtime_seconds']} s")


def write_report(m: dict, roas: pd.DataFrame, alloc: pd.DataFrame, conv: pd.DataFrame) -> None:
    b = m["budget"]
    lines = [
        "# Day 18 run report (SYNTHETIC data)",
        "",
        f"Data path: `{m['data']['path_used']}`; {m['data']['n_weeks']} weeks x {m['data']['n_regions']} regions; seed {m['data']['seed']}.",
        f"Sampler: NUTS, {m['sampler']['chains']} chains x {m['sampler']['draws']} draws ({m['sampler']['tune']} tune), {m['sampler']['fit_seconds']} s.",
        "",
        "## Convergence",
        "",
        f"- Max R-hat over all {m['convergence']['n_quantities']} posterior quantities: {m['convergence']['max_rhat']}",
        f"- Quantities with R-hat >= 1.01: {m['convergence']['n_rhat_ge_1_01']}",
        f"- Min bulk ESS: {m['convergence']['min_ess_bulk']}; min tail ESS: {m['convergence']['min_ess_tail']}; divergences: {m['convergence']['divergences']}",
        "",
        analysis.markdown_table(conv, 4),
        "",
        "## Predictive checks",
        "",
        f"- Prior predictive 94% interval covers {m['prior_predictive']['coverage_94_region_week']} of region-weeks; median interval width / observed = {m['prior_predictive']['interval_width_ratio']}",
        f"- Posterior predictive 94% interval covers {m['posterior_predictive']['coverage_94_region_week']} of region-weeks and {m['posterior_predictive']['coverage_94_national_week']} of national weeks",
        f"- Posterior mean R^2 (region-week) {m['posterior_predictive']['r2_region_week']}; MAPE {m['posterior_predictive']['mape_region_week']} (region-week), {m['posterior_predictive']['mape_national_week']} (national)",
        "",
        "## ROAS",
        "",
        analysis.markdown_table(roas, 3),
        "",
        "## Budget (annual $k)",
        "",
        analysis.markdown_table(alloc, 1),
        "",
    ]
    for k in ("expected_optimal", "risk_adjusted", "last_click_guided", "oracle_true_optimal"):
        v = b[k]
        lines.append(
            f"- **{k}**: posterior lift {v['lift_mean_k']:.1f} $k (94% HDI {v['lift_hdi_low_k']:.1f} to {v['lift_hdi_high_k']:.1f}), P(lift>0) {v['prob_lift_positive']:.3f}; true lift {v['true_lift_k']:.1f} $k"
        )
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
