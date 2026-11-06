import numpy as np

from pricing27.rm import bid_price, expected_sales, step_values, value_tables


def test_step_values_match_monte_carlo():
    rng = np.random.default_rng(0)
    prices = np.array([100.0, 150.0])
    lam = np.array([30.0, 12.0])
    v_next = np.sqrt(np.arange(41)) * 50
    R = 20
    ev = step_values(prices, lam, R, v_next)
    for c in range(2):
        n = rng.poisson(lam[c], 200_000)
        s = np.minimum(n, R)
        mc = np.mean(prices[c] * s + v_next[R - s])
        assert abs(mc - ev[c]) / ev[c] < 0.005
    assert np.allclose(
        expected_sales(lam, R),
        [np.mean(np.minimum(rng.poisson(lam[0], 200_000), R)), np.mean(np.minimum(rng.poisson(lam[1], 200_000), R))],
        rtol=0.01,
    )


def test_value_function_monotone_bellman_consistent_and_bid_prices():
    prices = [np.array([80.0, 120.0, 160.0, 200.0])] * 3
    lams = [np.array([40.0, 30.0, 20.0, 10.0])] * 3
    V = value_tables(prices, lams, capacity=150)
    assert len(V) == 4 and np.all(V[3] == 0)
    for j in range(3):
        assert np.all(np.diff(V[j]) >= -1e-9)  # more rooms never hurts
        # Bellman consistency between the vectorised backward induction and the one-step evaluator
        for R in (0, 1, 7, 33, 60, 150):
            assert np.isclose(V[j][R], step_values(prices[j], lams[j], R, V[j + 1]).max())
    bids = [bid_price(V[1], R) for R in range(1, 151)]
    assert all(b >= -1e-9 for b in bids) and bids[0] >= bids[-1]
    assert bids[-1] < 1.0  # plenty of rooms left -> near-zero opportunity cost
    assert bid_price(V[1], 0) == float("inf")


def test_scarcity_raises_optimal_price():
    prices = np.array([80.0, 120.0, 160.0, 200.0])
    lam = np.array([40.0, 30.0, 20.0, 10.0])
    V = value_tables([prices] * 2, [lam] * 2, capacity=100)
    best_many = prices[np.argmax(step_values(prices, lam, 100, V[1]))]
    best_few = prices[np.argmax(step_values(prices, lam, 8, V[1]))]
    assert best_few > best_many
