import math

import pytest
from statsmodels.stats.power import NormalIndPower

from expengine.power import mde_means, n_per_arm_means, n_per_arm_proportions, power_means


def test_n_matches_statsmodels():
    sigma, mde = 10.0, 0.5
    n = n_per_arm_means(sigma, mde)
    ref = NormalIndPower().solve_power(effect_size=mde / sigma, alpha=0.05, power=0.8, ratio=1.0)
    assert abs(n - math.ceil(ref)) <= 1


def test_cuped_reduction_scales_n():
    n0 = n_per_arm_means(10, 0.5)
    n1 = n_per_arm_means(10, 0.5, variance_reduction=0.5)
    assert abs(n1 / n0 - 0.5) < 0.01


def test_power_and_mde_are_consistent():
    n = n_per_arm_means(3.0, 0.2)
    assert power_means(3.0, 0.2, n) == pytest.approx(0.8, abs=0.01)
    assert mde_means(3.0, n) == pytest.approx(0.2, rel=0.01)


def test_proportions():
    n = n_per_arm_proportions(0.10, 0.10)
    assert 14_000 < n < 15_500  # textbook: ~14.7k per arm for 10% -> 11%
    assert n_per_arm_proportions(0.10, 0.10, variance_reduction=0.2) < n


@pytest.mark.parametrize(
    "kwargs",
    [dict(sigma=0, mde_abs=1), dict(sigma=1, mde_abs=0), dict(sigma=1, mde_abs=1, variance_reduction=1.0)],
)
def test_means_validation(kwargs):
    with pytest.raises(ValueError):
        n_per_arm_means(**kwargs)


def test_proportion_validation():
    with pytest.raises(ValueError):
        n_per_arm_proportions(0.0, 0.1)
    with pytest.raises(ValueError):
        n_per_arm_proportions(0.9, 0.2)
