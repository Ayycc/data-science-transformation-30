import numpy as np
import pytest
from scipy import stats

from expengine.cuped import (
    cuped_adjust,
    cuped_estimate,
    cuped_theta,
    diff_in_means,
    regression_adjusted,
    stats_from_sums,
)


def _data(rng, n=20_000, rho=0.8, effect=0.1):
    x = rng.standard_normal(n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    t = rng.random(n) < 0.5
    return y + effect * t, x, t


def test_naive_matches_welch(rng):
    y, _, t = _data(rng)
    est = diff_in_means(y, t)
    ref = stats.ttest_ind(y[t], y[~t], equal_var=False)
    assert est.p_value == pytest.approx(ref.pvalue, rel=0.02)
    assert est.ci_low < est.diff < est.ci_high


def test_cuped_variance_reduction_close_to_rho_squared(rng):
    y, x, t = _data(rng, rho=0.8)
    est = cuped_estimate(y, x, t)
    assert est.variance_reduction == pytest.approx(0.64, abs=0.03)
    assert est.diff == pytest.approx(0.1, abs=0.03)
    assert est.theta == pytest.approx(0.8, abs=0.03)


def test_cuped_matches_ols_ancova(rng):
    y, x, t = _data(rng)
    est = cuped_estimate(y, x, t)
    ols = regression_adjusted(y, x, t)
    assert est.diff == pytest.approx(ols["diff"], abs=2e-3)
    assert est.se == pytest.approx(ols["se"], rel=0.02)


def test_multivariate_cuped(rng):
    y, x, t = _data(rng)
    x2 = np.column_stack([x, rng.standard_normal(len(x))])
    est = cuped_estimate(y, x2, t)
    assert est.variance_reduction > 0.6
    theta = cuped_theta(y, x2)
    assert theta.shape == (2,) and abs(theta[1]) < 0.05
    assert regression_adjusted(y, x2, t)["diff"] == pytest.approx(est.diff, abs=5e-3)


def test_adjust_preserves_mean(rng):
    y, x, _ = _data(rng)
    assert cuped_adjust(y, x).mean() == pytest.approx(y.mean())


def test_constant_covariate_gives_naive(rng):
    y, _, t = _data(rng)
    est = cuped_estimate(y, np.ones_like(y), t)
    assert est.variance_reduction == pytest.approx(0.0, abs=1e-9)


def test_stats_from_sums_vectorised():
    c = {"n": np.array([10.0, 20.0]), "sy": np.array([10.0, 20.0]), "syy": np.array([20.0, 40.0])}
    t = {"n": np.array([10.0, 20.0]), "sy": np.array([20.0, 40.0]), "syy": np.array([50.0, 100.0])}
    s = stats_from_sums(c, t)
    assert np.allclose(s["diff"], [1.0, 1.0])
    assert s["var"][1] < s["var"][0]


def test_zero_variance_edge_case():
    y = np.array([1.0, 1.0, 1.0, 1.0])
    t = np.array([True, False, True, False])
    est = diff_in_means(y, t)
    assert est.se == 0 and est.p_value == 1.0
    est2 = diff_in_means(np.array([2.0, 1.0, 2.0, 1.0]), t)
    assert est2.p_value == 0.0
    assert "diff" in est2.to_dict()
