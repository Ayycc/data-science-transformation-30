from __future__ import annotations

import numpy as np

from retention import clv as clvm
from retention import policy as pol
from retention import uplift as up


def test_conditional_survival_and_clv():
    S = np.tile(0.9 ** np.arange(0, 20), (2, 1))  # constant 10% monthly hazard
    cond = clvm.conditional_survival(S, np.array([0, 5]), 6)
    assert np.allclose(cond, 0.9 ** np.arange(1, 7))  # memoryless
    ones = np.ones((1, 12))
    v = clvm.clv(ones, np.array([100.0]), 0.5, 0.0)
    assert np.isclose(v[0], 600.0)
    c = clvm.offer_cost(ones, np.array([100.0]), 2.0, 0.2, 3, 0.0)
    assert np.isclose(c[0], 2.0 + 60.0)
    mult = clvm.retention_value_multiplier(ones, np.array([100.0]), 0.5, 0.0, 6)
    assert np.isclose(mult[0], 50 * (sum(k / 6 for k in range(1, 6)) + 7))


def test_greedy_budget_and_value_policy():
    pr = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    cost = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    m = pol.greedy_budget(pr, cost, 25)
    assert m.tolist() == [True, True, False, False, False]
    net = np.array([5.0, -1.0, 3.0, 0.5, -2.0])
    v = pol.value_policy(net, cost, 100)
    assert v.tolist() == [True, False, True, True, False]


def test_qini_rewards_true_uplift():
    rng = np.random.default_rng(1)
    n = 40000
    tau = rng.uniform(-0.1, 0.3, n)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.5 + t * tau).astype(int)
    assert up.qini_coefficient(y, t, tau) > up.qini_coefficient(y, t, rng.random(n)) + 0.005
    f, g = up.true_gain_curve(tau, tau)
    assert np.isclose(g[-1], tau.mean())


def test_aipw_is_unbiased_with_wrong_outcome_model():
    rng = np.random.default_rng(2)
    n = 200000
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.6 + 0.1 * t).astype(float)
    g = up.aipw_scores(y, t, np.full(n, 0.3), np.full(n, 0.9), 0.5)  # deliberately wrong mu
    assert abs(g.mean() - 0.1) < 0.01
