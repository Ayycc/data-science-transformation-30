from __future__ import annotations

import numpy as np

from card_fraud import evaluate as ev

COST = {
    "true_positive": {"fixed": 5.0, "pct_of_amount": 0.0},
    "false_positive": {"fixed": 10.0, "pct_of_amount": 0.02},
    "false_negative": {"fixed": 20.0, "pct_of_amount": 1.0},
    "true_negative": {"fixed": 0.0, "pct_of_amount": 0.0},
}


def _data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.1
    score = np.round(np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1), 2)  # rounding creates ties
    amount = rng.lognormal(4, 1, n)
    return y, score, amount


def test_cost_curve_matches_brute_force() -> None:
    y, s, a = _data()
    cc = ev.cost_curve(y, s, a, COST)
    for thr, total, k in zip(cc["threshold"], cc["total_cost"], cc["n_alerts"]):
        r = ev.apply_threshold(y, s, a, COST, thr)
        assert r["alerts"] == k
        assert np.isclose(r["total_cost"], total)


def test_select_threshold_is_global_minimum() -> None:
    y, s, a = _data(seed=3)
    sel = ev.select_threshold(y, s, a, COST)
    grid = np.unique(np.concatenate([s, [np.inf]]))
    brute = min(ev.apply_threshold(y, s, a, COST, t)["total_cost"] for t in grid)
    assert np.isclose(sel["total_cost"], brute)


def test_approve_all_costs_all_fraud() -> None:
    y, s, a = _data()
    r = ev.apply_threshold(y, s, a, COST, np.inf)
    assert r["alerts"] == 0
    assert np.isclose(r["total_cost"], (20 + a[y]).sum())


def test_recall_at_fpr() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.4, 0.8, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert ev.recall_at_fpr(y, s, 0.0) == 0.5
    assert ev.recall_at_fpr(y, s, 0.125) == 1.0


def test_ece_perfect_and_bad() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = rng.random(20000) < p
    assert ev.expected_calibration_error(y, p) < 0.02
    assert ev.expected_calibration_error(y, np.clip(p * 0.2, 0, 1)) > 0.3
