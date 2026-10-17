import json

import numpy as np

from expengine.analyze import analyze_experiment, novelty_check
from expengine.montecarlo import evaluate, historical_baselines, population_truth, simulate_looks, tau2_for
from expengine.readout import render_readout
from expengine.simulate import EffectSpec, ExperimentSpec


def test_small_run_writes_everything(small_run):
    metrics, base = small_run
    res = base / "results"
    for f in [
        "metrics.json",
        "experiments_summary.csv",
        "monte_carlo_summary.csv",
        "hte_slices.csv",
        "cuped_variance_reduction.csv",
        "fpr_under_peeking.png",
        "cuped_variance_reduction.png",
        "novelty_curve.png",
        "sequential_traces.png",
    ]:
        assert (res / f).exists(), f
    assert len(list((res / "readouts").glob("E0*.md"))) == 6
    assert (base / "data" / "experiments.csv.gz").exists()
    m = json.loads((res / "metrics.json").read_text())
    assert set(m["acceptance"]) == {
        "sequential_fpr_under_peeking_le_5pct",
        "cuped_variance_reduction_ge_30pct_on_correlated_metrics",
    }
    assert m["data"]["source"].startswith("SYNTHETIC")
    assert "sessions" in metrics["monte_carlo"]["aa"]


def test_readout_flags_srm_and_novelty():
    tau2 = tau2_for(historical_baselines(20_000))
    from expengine.simulate import generate_experiment

    spec = ExperimentSpec("T1", "srm", "h", "converted", 40_000, 7, srm_drop_rate=0.4, seed=9)
    df, _, truth = generate_experiment(spec)
    r = analyze_experiment(df, spec, truth, tau2)
    assert r["srm"]["flagged"] and r["decision"].startswith("INVALID")
    md = render_readout(r)
    assert "FAILED" in md and "SYNTHETIC" in md and "dropped" in md

    spec = ExperimentSpec(
        "T2", "nov", "h", "sessions", 40_000, 7, EffectSpec(novelty_sessions=0.4, lift_sessions=0.05), seed=10
    )
    df, _, truth = generate_experiment(spec)
    r = analyze_experiment(df, spec, truth, tau2)
    assert r["novelty"]["flagged"]
    assert r["decision"].startswith("SHIP WITH CAUTION")
    assert "novelty effect detected" in render_readout(r)


def test_decisions_ship_and_harm():
    tau2 = tau2_for(historical_baselines(20_000))
    from expengine.simulate import generate_experiment

    spec = ExperimentSpec("T3", "win", "h", "sessions", 30_000, 7, EffectSpec(lift_sessions=0.08), seed=11)
    df, _, truth = generate_experiment(spec)
    assert analyze_experiment(df, spec, truth, tau2)["decision"].startswith("SHIP:")

    spec = ExperimentSpec("T4", "lose", "h", "sessions", 30_000, 7, EffectSpec(lift_sessions=-0.08), seed=12)
    df, _, truth = generate_experiment(spec)
    assert analyze_experiment(df, spec, truth, tau2)["decision"].startswith("DO NOT SHIP: primary")

    spec = ExperimentSpec(
        "T5", "guard", "h", "revenue", 30_000, 7, EffectSpec(lift_basket=0.6, lift_sessions=-0.08), seed=13
    )
    df, _, truth = generate_experiment(spec)
    r = analyze_experiment(df, spec, truth, tau2)
    assert "guardrail" in r["decision"] and "sessions" in r["guardrail_harm"]


def test_novelty_check_null(sample_df):
    df = sample_df.assign(treat=sample_df.variant.eq("treatment"))
    out = novelty_check(df)
    assert not out["flagged"]


def test_monte_carlo_small():
    tau2 = tau2_for(historical_baselines(20_000))
    sims = simulate_looks(EffectSpec(), 150, 5, 200, seed=1, max_users_per_chunk=20_000)
    ev = evaluate(sims, tau2)
    s = ev["sessions"]
    assert s["n_reps"] == 150 and s["n_looks"] == 5
    assert s["rejection_rate"]["msprt"] <= s["rejection_rate"]["naive_peeking"]
    assert s["cuped_variance_reduction_empirical"] > 0.45
    truth = population_truth(EffectSpec(lift_sessions=0.05), 20_000)
    ev2 = evaluate(simulate_looks(EffectSpec(lift_sessions=0.05), 20, 5, 200, seed=2), tau2, truth=truth)
    assert 0 <= ev2["sessions"]["always_valid_ci_coverage_all_looks"] <= 1
    assert np.isfinite(ev2["sessions"]["bias_cuped_final"])
