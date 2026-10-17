import math

import pytest
from scipy import stats

from expengine.bayes import bayes_decision, beta_binomial, normal_posterior


def test_beta_binomial_agrees_with_normal_approx():
    r = beta_binomial(1000, 10_000, 1100, 10_000, seed=1)
    se = math.sqrt(0.1 * 0.9 / 10_000 + 0.11 * 0.89 / 10_000)
    assert r["prob_t_better"] == pytest.approx(stats.norm.cdf(0.01 / se), abs=0.01)
    assert r["expected_loss_t"] < r["expected_loss_c"]
    assert r["rel_lift_ci_low"] < 0.1 < r["rel_lift_ci_high"]


def test_normal_posterior_flat_and_shrunk():
    flat = normal_posterior(0.2, 0.1)
    assert flat["post_mean"] == pytest.approx(0.2)
    assert flat["prob_positive"] == pytest.approx(stats.norm.cdf(2.0))
    shrunk = normal_posterior(0.2, 0.1, prior_mean=0.0, prior_sd=0.1)
    assert shrunk["post_mean"] == pytest.approx(0.1)
    assert shrunk["post_sd"] < 0.1
    assert flat["expected_loss_t"] < flat["expected_loss_c"]
    with pytest.raises(ValueError):
        normal_posterior(0.1, 0.0)


def test_expected_loss_closed_form(rng):
    m, s = 0.05, 0.1
    draws = rng.normal(m, s, 400_000)
    r = normal_posterior(m, s)
    assert r["expected_loss_t"] == pytest.approx((-draws).clip(0).mean(), rel=0.02)


def test_decision_rule():
    assert bayes_decision(0.97, 0.0001, 0.001) == "ship"
    assert bayes_decision(0.97, 0.01, 0.001).startswith("keep")
    assert bayes_decision(0.02, 0.01, 0.001) == "do not ship"
