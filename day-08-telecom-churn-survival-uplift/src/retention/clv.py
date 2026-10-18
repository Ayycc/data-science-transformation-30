"""Customer lifetime value and churn probabilities from fitted survival curves."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def survival_grid(cph: CoxPHFitter, X: pd.DataFrame, t_max_fit: int, t_needed: int, slope_months: int = 4) -> np.ndarray:
    """S_i(t) for t = 0..t_needed (columns). Beyond the last fitted month the cumulative hazard is
    extended linearly with each customer's average monthly hazard over the last ``slope_months``."""
    times = np.arange(0, t_max_fit + 1, dtype=float)
    S = cph.predict_survival_function(X, times=times)[X.index].to_numpy().T
    S = np.clip(S, 1e-12, 1.0)
    H = -np.log(S)
    if t_needed > t_max_fit:
        slope = (H[:, -1] - H[:, -1 - slope_months]) / slope_months
        extra = np.arange(1, t_needed - t_max_fit + 1)
        H = np.hstack([H, H[:, -1:] + slope[:, None] * extra[None, :]])
    return np.exp(-H)


def conditional_survival(S: np.ndarray, t0: np.ndarray, horizon: int) -> np.ndarray:
    """S_i(t0+k)/S_i(t0) for k = 1..horizon -> (n, horizon)."""
    rows = np.arange(S.shape[0])[:, None]
    cols = t0[:, None] + np.arange(1, horizon + 1)[None, :]
    return S[rows, cols] / S[np.arange(S.shape[0]), t0][:, None]


def discount_factors(horizon: int, monthly_rate: float) -> np.ndarray:
    return (1 + monthly_rate) ** -np.arange(1, horizon + 1)


def clv(cond_s: np.ndarray, monthly_charges: np.ndarray, margin_rate: float, monthly_rate: float) -> np.ndarray:
    """Expected discounted margin over the horizon: sum_k S(k) * margin * charge / (1+r)^k."""
    d = discount_factors(cond_s.shape[1], monthly_rate)
    return (cond_s * d).sum(1) * margin_rate * monthly_charges


def retention_value_multiplier(
    cond_s: np.ndarray, monthly_charges: np.ndarray, margin_rate: float, monthly_rate: float, window: int
) -> np.ndarray:
    """Dollar value of +1 in 6-month retention probability.

    A customer saved from churning inside the window contributes, on average, part of the window
    (saves are spread over months 1..w, so month k is gained with weight k/w) plus the discounted
    margin of a survivor from the end of the window, using the survival curve conditional on
    being alive at the end of the window.
    """
    h = cond_s.shape[1]
    d = discount_factors(h, monthly_rate)
    k = np.arange(1, h + 1)
    within = np.where(k < window, k / window, 0.0)
    after = np.where(k >= window, cond_s / cond_s[:, [window - 1]], 0.0)
    return margin_rate * monthly_charges * ((within + after) * d).sum(1)


def offer_cost(
    cond_s: np.ndarray,
    monthly_charges: np.ndarray,
    contact_cost: float,
    discount_share: float,
    discount_months: int,
    monthly_rate: float,
) -> np.ndarray:
    """Expected cost of contacting a customer: contact fee + discount paid while they stay."""
    d = discount_factors(discount_months, monthly_rate)
    return contact_cost + discount_share * monthly_charges * (cond_s[:, :discount_months] * d).sum(1)
