import numpy as np

from pricing27.policies import LinUCB, Thompson, _Ridge
from pricing27.simulator import make_world
from pricing27.study import StudyConfig, make_policies, regret_exponent, run_policy


def test_ridge_recovers_coefficients_and_noise():
    rng = np.random.default_rng(0)
    theta = rng.normal(size=5)
    r = _Ridge(5, lam=1e-3, sigma0=1.0)
    for _ in range(3000):
        x = rng.normal(size=5)
        r.update(x, float(x @ theta + rng.normal(0, 0.3)))
    assert np.allclose(r.theta, theta, atol=0.03)
    assert abs(r.sigma - 0.3) < 0.03


def test_warm_start_uses_only_randomised_cells(logs):
    b = LinUCB()
    n = b.warm_start(logs)
    assert n == int(logs.price_test.sum()) == sum(m.n for m in b.models)


def test_bandits_learn_regret_falls_and_beats_static(cfg, models):
    scfg = StudyConfig(n_seeds=1, n_nights=730)
    w = make_world(365, 730, 21, cfg)
    u = np.random.default_rng(3).random((730, 3))
    pols = make_policies(("static", "thompson", "linucb"), models, cfg, scfg)
    res = {n: run_policy(p, w, cfg, scfg.guard, models, u, 1) for n, p in pols.items()}
    for n in ("thompson", "linucb"):
        assert res[n]["cum_regret"][-1] < 0.5 * res["static"]["cum_regret"][-1]
        assert res[n]["revenue"] > res["static"]["revenue"]
        cum = res[n]["cum_regret"]
        assert cum[-1] - cum[364] < cum[364]  # year-2 regret below year-1 regret (same seasons)
    assert isinstance(pols["thompson"], Thompson)


def test_regret_exponent_detects_linear_and_sqrt():
    t = np.arange(1, 1001)
    assert abs(regret_exponent(5.0 * t) - 1.0) < 1e-6
    assert abs(regret_exponent(5.0 * np.sqrt(t)) - 0.5) < 1e-6
