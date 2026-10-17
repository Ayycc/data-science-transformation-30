import pytest

from expengine.srm import srm_test


def test_balanced_passes():
    r = srm_test([50_100, 49_900], [0.5, 0.5])
    assert not r.flagged and r.p_value > 0.1
    assert r.observed_share[0] == pytest.approx(0.501)


def test_small_imbalance_at_scale_is_flagged():
    r = srm_test([50_800, 49_200], [0.5, 0.5])
    assert r.flagged and r.p_value < 1e-6


def test_unequal_design():
    assert not srm_test([20_000, 80_000], [1, 4]).flagged
    assert srm_test([25_000, 75_000], [1, 4]).flagged


@pytest.mark.parametrize("obs,ratios", [([1, 2], [1]), ([5], [1]), ([5, 5], [0, 1])])
def test_validation(obs, ratios):
    with pytest.raises(ValueError):
        srm_test(obs, ratios)
