import numpy as np

from pricing27.simulator import (
    competitor_price,
    context_for,
    is_weekend,
    make_world,
    sample_demand,
    shock_posterior,
    static_rate_table,
    true_demand_given_u,
    true_expected_demand,
)


def test_static_table_is_seasonal_rounded_and_has_weekend_uplift(cfg):
    days = np.arange(365)
    rates = static_rate_table(days, cfg)
    wk = is_weekend(days, cfg) == 1
    base = rates / np.where(wk, 1.10, 1.0)
    assert np.allclose(base / 5, np.round(base / 5))
    assert rates[196] > rates[15]  # July above January
    assert wk.sum() == 104


def test_demand_decreases_in_price_and_rises_with_shock(cfg):
    w = make_world(0, 10, 1, cfg)
    ctx = context_for(w, 3, 1, competitor_price(w, 3))
    p = ctx.static * np.linspace(0.6, 1.8, 13)
    d = true_expected_demand(ctx, p, cfg)
    assert np.all(np.diff(d) < 0)
    assert true_demand_given_u(ctx, p[:1], 1.0, cfg)[0] > true_demand_given_u(ctx, p[:1], -1.0, cfg)[0]


def test_shock_posterior_moves_towards_high_u_after_high_demand(cfg):
    w = make_world(0, 10, 1, cfg)
    ctx = context_for(w, 2, 0, competitor_price(w, 2))
    price = ctx.static
    mean = float(true_expected_demand(ctx, np.array([price]), cfg)[0])
    from pricing27.simulator import _GH_X

    hi = shock_posterior([(ctx, price, int(mean * 1.6))], cfg)
    lo = shock_posterior([(ctx, price, int(mean * 0.5))], cfg)
    assert np.isclose(hi.sum(), 1) and np.isclose(lo.sum(), 1)
    assert hi @ _GH_X > 0.5 > -0.5 > lo @ _GH_X


def test_sampling_is_seeded(cfg):
    w = make_world(0, 5, 3, cfg)
    ctx = context_for(w, 0, 0, competitor_price(w, 0))
    a = [sample_demand(ctx, 150.0, 0.0, cfg, np.random.default_rng(9)) for _ in range(2)]
    assert a[0] == a[1]


def test_logs_reproducible_and_small(cfg, logs):
    from pricing27.estimation import generate_logs

    again = generate_logs(cfg, seed=20261106)
    assert again.equals(logs)
    assert len(logs) == 365 * 3
    assert (logs.sold <= logs.rooms_left_before).all()
    assert logs.groupby("night").sold.sum().max() <= cfg.capacity
