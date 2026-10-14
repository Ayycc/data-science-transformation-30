"""Metric implementations against known answers."""

from __future__ import annotations

import numpy as np

from readmit.evaluate import calibration_slope_intercept, kaplan_meier, net_benefit


def _sim(n: int = 40_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lp = rng.normal(-1.6, 1.0, n)
    p = 1 / (1 + np.exp(-lp))
    return rng.binomial(1, p), lp


def test_calibration_slope_one_when_calibrated() -> None:
    y, lp = _sim()
    s, c = calibration_slope_intercept(y, 1 / (1 + np.exp(-lp)))
    assert abs(s - 1) < 0.05 and abs(c) < 0.05


def test_calibration_slope_detects_overfitting() -> None:
    y, lp = _sim()
    s, _ = calibration_slope_intercept(y, 1 / (1 + np.exp(-2 * lp)))  # too extreme predictions
    assert abs(s - 0.5) < 0.05


def test_calibration_in_the_large_detects_offset() -> None:
    y, lp = _sim()
    _, c = calibration_slope_intercept(y, 1 / (1 + np.exp(-(lp - 0.5))))  # under-predicts
    assert abs(c - 0.5) < 0.05


def test_net_benefit_by_hand() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.3, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # at 0.25: treat 3 (2 TP, 1 FP) -> 2/10 - 1/10 * 0.25/0.75
    assert np.isclose(net_benefit(y, p, np.array([0.25]))[0], 0.2 - 0.1 / 3)


def test_kaplan_meier_simple() -> None:
    km = kaplan_meier(np.array([2, 5, 30, 30]), np.array([1, 1, 0, 0]), horizon=30)
    assert np.isclose(km.loc[km.day == 2, "survival"].iloc[0], 0.75)
    assert np.isclose(km.loc[km.day == 30, "cumulative_incidence"].iloc[0], 0.5)
