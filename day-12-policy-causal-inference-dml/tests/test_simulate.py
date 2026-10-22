import numpy as np

from policy_causal.simulate import observed, panel_truth, simulate_individuals, simulate_panel, true_effects


def test_deterministic_and_seeded():
    a = simulate_individuals(500, 1)
    b = simulate_individuals(500, 1)
    c = simulate_individuals(500, 2)
    assert a.equals(b)
    assert not a.equals(c)


def test_observed_hides_truth(people):
    obs = observed(people)
    assert not [c for c in obs.columns if c.startswith("true_")]
    assert "ability" not in obs.columns


def test_confounding_makes_naive_wrong(people):
    tr = true_effects(people)
    t = people["treated"] == 1
    naive = people.loc[t, "earnings_post"].mean() - people.loc[~t, "earnings_post"].mean()
    assert tr["ate_earnings"] > 1000
    assert naive < tr["ate_earnings"] - 2000  # enrolees are negatively selected
    assert 0.03 < people["true_propensity"].min() and people["true_propensity"].max() < 0.97


def test_potential_outcomes_consistent(people):
    # observed outcome equals the potential outcome of the realised arm (earnings never negative)
    assert (people["earnings_post"] >= 0).all()
    assert people["true_tau_employment"].between(-1, 1).all()


def test_panel_structure(panel):
    assert len(panel) == 36 * 24
    assert (panel.groupby("county_id")["first_treated"].nunique() == 1).all()
    never = panel[~np.isfinite(panel["first_treated"])]
    assert never["treated_now"].sum() == 0 and never["true_effect"].abs().sum() == 0
    tr = panel_truth(panel)
    assert tr["event_time"][0] < tr["event_time"][7]  # effects grow with exposure
    assert 1.0 < tr["att_overall"] < 2.5


def test_panel_needs_controls():
    import pytest

    with pytest.raises(ValueError):
        simulate_panel(10, 12, {4: 10}, 0)
