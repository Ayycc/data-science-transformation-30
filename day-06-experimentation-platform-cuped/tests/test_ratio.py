import numpy as np
import pytest

from expengine.ratio import ratio_mean_var, ratio_metric_test, relative_lift


def test_delta_variance_matches_bootstrap(rng):
    n = 4000
    den = rng.poisson(5, n) + 1.0
    num = den * rng.gamma(2.0, 1.5, n)
    r, var = ratio_mean_var(num, den)
    assert r == pytest.approx(num.sum() / den.sum())
    boots = []
    for _ in range(400):
        i = rng.integers(0, n, n)
        boots.append(num[i].sum() / den[i].sum())
    assert np.sqrt(var) == pytest.approx(np.std(boots), rel=0.12)


def test_ratio_test_detects_effect(rng):
    n = 20_000
    t = rng.random(n) < 0.5
    den = rng.poisson(5, n) + 1.0
    num = den * rng.gamma(2.0, 1.5, n) * (1 + 0.1 * t)
    res = ratio_metric_test(num, den, t)
    assert res["p_value"] < 1e-4
    assert res["rel_ci_low"] < 0.1 < res["rel_ci_high"]


def test_relative_lift():
    rel, lo, hi = relative_lift(1.1, 0.0001, 1.0, 0.0001)
    assert rel == pytest.approx(0.1) and lo < 0.1 < hi
    assert np.isnan(relative_lift(1.0, 0.1, 0.0, 0.1)[0])


def test_ratio_validation():
    with pytest.raises(ValueError):
        ratio_mean_var(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        ratio_mean_var(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
