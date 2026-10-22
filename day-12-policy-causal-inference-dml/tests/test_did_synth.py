import numpy as np

from policy_causal.did import callaway_santanna, twfe
from policy_causal.simulate import panel_truth, simulate_panel
from policy_causal.synth import sc_weights, synthetic_control


def test_cs_exact_without_noise():
    p = simulate_panel(30, 20, {6: 8, 10: 8, 14: 8}, 3, noise_sd=0.0)
    # remove the factor structure too: equalise loadings by differencing out never-treated mean trend
    p["employment_rate"] = 70 + p["true_effect"] + 0.1 * p["quarter"] + p["county_id"] * 0.01
    r = callaway_santanna(p, n_boot=19, seed=0)
    assert abs(r["att_overall"] - panel_truth(p)["att_overall"]) < 1e-9
    assert r["pretrend_max_abs"] < 1e-9


def test_cs_covers_truth_and_beats_twfe(panel):
    tr = panel_truth(panel)["att_overall"]
    r = callaway_santanna(panel, n_boot=99, seed=0)
    assert r["att_overall_ci"][0] - 0.2 <= tr <= r["att_overall_ci"][1] + 0.2
    tw = twfe(panel)
    assert tw["estimate"] < tr  # TWFE is biased down with growing, staggered effects
    assert abs(r["att_overall"] - tr) < abs(tw["estimate"] - tr)


def test_sc_weights_simplex():
    rng = np.random.default_rng(0)
    donors = rng.normal(size=(10, 5))
    target = donors @ np.array([0.5, 0.5, 0, 0, 0])
    w = sc_weights(target, donors)
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()
    assert np.allclose(w[:2], 0.5, atol=1e-3)


def test_synthetic_control_runs(panel):
    first = panel.groupby("county_id")["first_treated"].first()
    unit = int(first[first == 8].index[0])
    r = synthetic_control(panel, unit)
    assert r["avg_post_gap"] > 0.5
    assert 0 < r["placebo_p_value"] <= 1
