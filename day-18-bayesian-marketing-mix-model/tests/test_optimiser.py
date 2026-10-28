import numpy as np

from bayes_mmm.optimiser import ResponseModel, linear_allocation, objective, optimise, summarise_lift


def _rm(seed=0, s=200):
    rng = np.random.default_rng(seed)
    beta = np.exp(rng.normal(np.log([0.10, 0.04, 0.08, 0.05]), 0.2, (s, 3, 4)))
    lam = np.exp(rng.normal(np.log([2.0, 3.0, 1.5, 2.5]), 0.2, (s, 4)))
    return ResponseModel(beta=beta, lam=lam, population=np.array([1.0, 2.0, 0.5]), y_scale=110.0, channel_scale=np.array([10.0, 6.0, 8.0, 5.0]))


def test_gradient_matches_finite_difference():
    rm = _rm()
    s = np.array([3.0, 2.0, 1.5, 2.0])
    for k in (0.0, 1.0):
        f, g = objective(rm, s, k)
        eps = 1e-6
        num = np.array([(objective(rm, s + eps * np.eye(4)[i], k)[0] - f) / eps for i in range(4)])
        np.testing.assert_allclose(g, num, rtol=1e-3, atol=1e-6)


def test_optimiser_respects_budget_and_bounds_and_improves():
    rm = _rm()
    s0 = np.array([3.0, 2.0, 1.5, 2.0])
    lo, hi = 0.5 * s0, 1.5 * s0
    for k in (0.0, 1.0):
        s = optimise(rm, s0, lo, hi, risk_aversion=k, n_starts=4)
        assert abs(s.sum() - s0.sum()) < 1e-6
        assert np.all(s >= lo - 1e-9) and np.all(s <= hi + 1e-9)
        assert -objective(rm, s, k)[0] >= -objective(rm, s0, k)[0] - 1e-9
    lift = summarise_lift(rm, optimise(rm, s0, lo, hi), s0)
    assert lift["lift_mean_k"] >= 0 and lift["lift_hdi_low_k"] <= lift["lift_mean_k"] <= lift["lift_hdi_high_k"]


def test_risk_aversion_reduces_spread():
    rm = _rm()
    s0 = np.array([3.0, 2.0, 1.5, 2.0])
    lo, hi = 0.5 * s0, 1.5 * s0
    a = optimise(rm, s0, lo, hi, 0.0, n_starts=4)
    b = optimise(rm, s0, lo, hi, 2.0, n_starts=4)
    assert rm.draws(b).std() <= rm.draws(a).std() + 1e-9


def test_linear_allocation_fills_best_channel():
    s0 = np.array([1.0, 1.0, 1.0, 1.0])
    s = linear_allocation(np.array([0.1, 5.0, 0.2, 0.3]), s0, 0.5 * s0, 1.5 * s0)
    assert s[1] == 1.5 and abs(s.sum() - 4.0) < 1e-9
