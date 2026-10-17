"""End-to-end run: simulate the portfolio, analyse and write readouts, run the Monte Carlo study."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from expengine import plots
from expengine.analyze import analyze_experiment
from expengine.montecarlo import TAU_REL, evaluate, historical_baselines, population_truth, simulate_looks, tau2_for
from expengine.readout import render_readout
from expengine.simulate import COVARIATE, METRICS, EffectSpec, default_portfolio, generate_experiment

MC_EFFECT = EffectSpec(lift_sessions=0.02, lift_purchase=0.06, lift_basket=0.0)


@dataclass(frozen=True)
class RunConfig:
    name: str
    portfolio_scale: float
    mc_reps: int
    mc_looks: int
    mc_users_per_look: int
    baseline_n: int
    truth_n: int
    mc_effect: EffectSpec = field(default_factory=lambda: MC_EFFECT)
    seed: int = 2026


FULL = RunConfig("full", 1.0, mc_reps=2000, mc_looks=21, mc_users_per_look=1000, baseline_n=200_000, truth_n=400_000)
SMALL = RunConfig("small", 0.05, mc_reps=60, mc_looks=7, mc_users_per_look=200, baseline_n=20_000, truth_n=20_000)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating | float):
        f = float(o)
        return None if not np.isfinite(f) else f
    return o


def run(cfg: RunConfig, results_dir: Path, data_dir: Path | None = None, log: Callable[[str], None] = print) -> dict:
    t0 = time.time()
    results_dir = Path(results_dir)
    (results_dir / "readouts").mkdir(parents=True, exist_ok=True)
    if data_dir is not None:
        Path(data_dir).mkdir(parents=True, exist_ok=True)

    baselines = historical_baselines(cfg.baseline_n, seed=7)
    tau2 = tau2_for(baselines)
    log(f"historical baselines: {baselines}")

    readouts, curves, frames = [], {}, []
    for spec in default_portfolio(cfg.portfolio_scale):
        df, curve, truth = generate_experiment(spec)
        frames.append(df)
        curves[spec.exp_id] = curve
        r = analyze_experiment(df, spec, truth, tau2, seed=cfg.seed)
        readouts.append(r)
        (results_dir / "readouts" / f"{spec.exp_id}.md").write_text(render_readout(r))
        log(f"{spec.exp_id} {spec.name}: {r['decision']}")
    if data_dir is not None:
        pd.concat(frames).to_csv(Path(data_dir) / "experiments.csv.gz", index=False)
        pd.concat(curves.values()).to_csv(Path(data_dir) / "exposure_day_curves.csv", index=False)

    # Monte Carlo
    log(f"Monte Carlo A/A: {cfg.mc_reps} reps x {cfg.mc_looks} looks x {cfg.mc_users_per_look} users/arm/look")
    aa = evaluate(simulate_looks(EffectSpec(), cfg.mc_reps, cfg.mc_looks, cfg.mc_users_per_look, seed=cfg.seed), tau2)
    truth = population_truth(cfg.mc_effect, cfg.truth_n)
    log("Monte Carlo with a true effect")
    eff = evaluate(
        simulate_looks(cfg.mc_effect, cfg.mc_reps, cfg.mc_looks, cfg.mc_users_per_look, seed=cfg.seed + 1),
        tau2,
        truth=truth,
    )

    # correlation of each metric with its covariate, from the pooled portfolio (all six experiments)
    pooled = pd.concat(frames)
    rho = {m: float(np.corrcoef(pooled[m], pooled[COVARIATE[m]])[0, 1]) for m in METRICS}

    # tables
    summary_rows = []
    for r in readouts:
        pm = r["primary_metric"]
        mm = r["metrics"][pm]
        summary_rows.append(
            {
                "exp_id": r["exp_id"],
                "name": r["name"],
                "primary_metric": pm,
                "n_users": r["n_control"] + r["n_treatment"],
                "srm_p": r["srm"]["p_value"],
                "srm_flagged": r["srm"]["flagged"],
                "true_rel_lift": mm["true_rel"],
                "naive_rel_lift": mm["naive"]["rel_lift"],
                "cuped_rel_lift": mm["cuped"]["rel_lift"],
                "cuped_rel_ci_low": mm["cuped"]["rel_ci_low"],
                "cuped_rel_ci_high": mm["cuped"]["rel_ci_high"],
                "variance_reduction": mm["cuped"]["variance_reduction"],
                "naive_p": mm["naive"]["p_value"],
                "always_valid_p": mm["always_valid_p_final"],
                "first_significant_day": mm["first_reject_day"],
                "first_naive_significant_day": mm["first_naive_reject_day"],
                "bayes_decision": r["bayes"]["decision"],
                "novelty_flagged": r["novelty"]["flagged"],
                "late_rel_lift": r["novelty"]["late_rel_lift"],
                "n_per_arm_naive": r["power"]["n_per_arm_naive"],
                "n_per_arm_cuped": r["power"]["n_per_arm_cuped"],
                "cuped_ci_covers_truth": mm["cuped_ci_covers_truth"],
                "decision": r["decision"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_dir / "experiments_summary.csv", index=False)

    hte_rows = [dict(exp_id=r["exp_id"], **h) for r in readouts for h in r["hte"]]
    pd.DataFrame(hte_rows).to_csv(results_dir / "hte_slices.csv", index=False)

    mc_rows = []
    for scen, res in (("A/A (no effect)", aa), ("true effect", eff)):
        for m, e in res.items():
            row = {"scenario": scen, "metric": m, **{f"reject_{k}": v for k, v in e["rejection_rate"].items()}}
            row["vr_empirical"] = e["cuped_variance_reduction_empirical"]
            row["median_stop_msprt"] = e["median_stopping_look"]["msprt"]
            row["median_stop_msprt_cuped"] = e["median_stopping_look"]["msprt_cuped"]
            row["always_valid_ci_coverage"] = e.get("always_valid_ci_coverage_all_looks")
            row["fixed_ci_coverage"] = e.get("fixed_ci_coverage_final")
            mc_rows.append(row)
    pd.DataFrame(mc_rows).to_csv(results_dir / "monte_carlo_summary.csv", index=False)

    cuped_rows = pd.DataFrame(
        [{"metric": m, "rho": rho[m], "vr_empirical": aa[m]["cuped_variance_reduction_empirical"]} for m in METRICS]
    )
    cuped_rows.to_csv(results_dir / "cuped_variance_reduction.csv", index=False)

    # figures
    plots.fpr_curves(aa, results_dir / "fpr_under_peeking.png")
    plots.cuped_bars(cuped_rows, results_dir / "cuped_variance_reduction.png")
    novelty = next((r for r in readouts if r["exp_id"] == "E04"), readouts[0])
    plots.novelty_curve(
        curves[novelty["exp_id"]],
        results_dir / "novelty_curve.png",
        f"{novelty['exp_id']} {novelty['name']}: lift by exposure day (SYNTHETIC)",
    )
    plots.sequential_traces(
        [r for r in readouts if r["exp_id"] in ("E01", "E03", "E06")], results_dir / "sequential_traces.png"
    )

    max_fpr_seq = max(max(aa[m]["rejection_rate"]["msprt"], aa[m]["rejection_rate"]["msprt_cuped"]) for m in METRICS)
    correlated = [m for m in METRICS if rho[m] >= 0.55]
    acceptance = {
        "sequential_fpr_under_peeking_le_5pct": {
            "max_fpr_any_metric": max_fpr_seq,
            "passed": bool(max_fpr_seq <= 0.05),
        },
        "cuped_variance_reduction_ge_30pct_on_correlated_metrics": {
            "correlated_metrics_rho_ge_0.55": correlated,
            "vr": {m: aa[m]["cuped_variance_reduction_empirical"] for m in METRICS},
            "passed": bool(correlated) and all(aa[m]["cuped_variance_reduction_empirical"] >= 0.30 for m in correlated),
        },
    }

    metrics = {
        "run": {"config": asdict(cfg), "seconds": round(time.time() - t0, 1)},
        "data": {
            "source": (
                "SYNTHETIC (expengine.simulate, seeded); no public user-level experiment data with known ground truth"
            ),
            "n_experiments": len(readouts),
            "n_users_total": int(sum(r["n_control"] + r["n_treatment"] for r in readouts)),
        },
        "historical_baselines": baselines,
        "tau_rel": TAU_REL,
        "tau2": tau2,
        "covariate_correlation": rho,
        "experiments": {r["exp_id"]: {k: v for k, v in r.items() if k != "trace"} for r in readouts},
        "monte_carlo": {"aa": aa, "effect": eff, "effect_spec": asdict(cfg.mc_effect), "true_effect": truth},
        "acceptance": acceptance,
    }
    (results_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2))
    log(f"done in {time.time() - t0:.1f}s")
    return metrics
