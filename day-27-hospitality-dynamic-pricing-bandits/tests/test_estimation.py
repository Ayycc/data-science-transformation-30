import numpy as np

from pricing27.estimation import two_sls


def test_two_sls_recovers_slope_under_endogeneity():
    rng = np.random.default_rng(0)
    n = 20_000
    u = rng.normal(size=n)
    z = rng.choice([0.8, 1.0, 1.2], n)
    x = rng.normal(size=n)
    m = z + 0.3 * u + 0.1 * x
    y = 1.0 + 0.5 * x - 2.0 * m + u
    exog = np.column_stack([np.ones(n), x])
    ols = np.linalg.lstsq(np.column_stack([exog, m]), y, rcond=None)[0][-1]
    iv = two_sls(y, m, exog, z)
    assert ols > -1.0  # badly biased
    assert abs(iv["beta"][-1] + 2.0) < 4 * iv["se"][-1]
    assert iv["first_stage_F"] > 100


def test_naive_estimate_confounded_iv_and_randomised_cover_truth(est):
    truth = est["truth"]["theta_mean"]
    assert est["naive_ols"]["theta"] > truth + 1.0  # price follows demand -> slope biased towards zero
    for k in ("iv_2sls", "randomised_test_ols"):
        th, se = est[k]["theta"], est[k]["se"]
        assert th - 1.96 * se <= truth <= th + 1.96 * se
    assert est["iv_2sls"]["first_stage_F"] > 100


def test_semilog_model_predicts_positive_decreasing_demand(models, cfg):
    from pricing27.simulator import competitor_price, context_for, make_world

    w = make_world(400, 5, 2, cfg)
    ctx = context_for(w, 1, 0, competitor_price(w, 1))
    p = ctx.static * np.linspace(0.6, 1.8, 13)
    d = models["iv_2sls"].predict(ctx, p)
    assert np.all(d >= 0) and np.all(np.diff(d) < 0)
