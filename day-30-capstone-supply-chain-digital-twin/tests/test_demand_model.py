import numpy as np

from twin30 import demand


def test_design_fourier_has_expected_shape():
    dow, four, years = demand.design(0, 14)
    assert four.shape == (14, 2 * demand.K_FOURIER)
    assert dow[0] == 0 and years[7] == 7 / 364


def test_posterior_mean_matches_truth_for_oracle(small_run, oracle_posterior):
    from twin30 import world

    mu = oracle_posterior.mean(0, 0, 112, small_run.promo, small_run.post)
    truth = world.true_mean(0, 112, small_run.promo, small_run.post, small_run.truth)
    assert np.allclose(mu, truth, rtol=1e-9)


def test_naive_promo_lift_is_a_ratio(small_tables):
    lift = demand.naive_promo_lift(small_tables["sales"], 0, 112)
    assert lift.shape == (3,) and np.all(np.isfinite(lift) | np.isnan(lift))


def test_hdi_contains_the_mode_of_a_normal():
    x = np.random.default_rng(0).normal(size=5000)
    lo, hi = demand.hdi(x, 0.9)
    assert lo < 0 < hi and abs(hi - lo - 2 * 1.645) < 0.15


def test_pymc_smoke_fit_recovers_promo_direction():
    """Tiny NUTS run on a short SYNTHETIC world; checks plumbing and the sign of the promo effect."""
    from twin30 import world

    run = world.simulate_world(seed=8, n_days=364)
    sales = world.history_tables(run)["sales"]
    post, diag = demand.fit(sales, 0, 364, draws=60, tune=60, chains=1, max_rows=1200, seed=2)
    assert post.a.shape[1:] == (12, 3) and post.n == 60
    assert diag["divergences"] < 30
    assert (np.exp(post.promo).mean(axis=0) > 1.05).all()
