"""Inventory control: finite-horizon dynamic programme over the booking windows of one night.

For a stay night with booking windows j = 0..K-1, ``V_j(R)`` is the expected revenue still to
come with ``R`` rooms left at the start of window j. Demand in window j at price p is Poisson
with mean lambda_j(p) (unconstrained demand), sales are min(N, R):

    V_j(R) = max_p E[ p * min(N, R) + V_{j+1}(R - min(N, R)) ],   V_K = 0.

The bid price (opportunity cost of one room) entering window j is V_{j+1}(R) - V_{j+1}(R-1).
Pricing a window by maximising the one-step expression with a given V_{j+1} is DP control;
a bandit plugs its own demand estimate into the same expression.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import gammaln

_GLN = gammaln(np.arange(0, 2001) + 1.0)


def poisson_pmf_matrix(lam: np.ndarray, n_max: int) -> np.ndarray:
    """pmf[c, n] for n = 0..n_max-1, one row per mean in ``lam``."""
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-9)[:, None]
    n = np.arange(n_max)[None, :]
    return np.exp(n * np.log(lam) - lam - _GLN[:n_max][None, :])


def step_values(prices: np.ndarray, lam: np.ndarray, R: int, v_next: np.ndarray) -> np.ndarray:
    """E[p*min(N,R) + V_next(R - min(N,R))] for each candidate price (vector over candidates)."""
    prices = np.asarray(prices, dtype=float)
    if R <= 0:
        return np.full(len(prices), float(v_next[0]))
    pmf = poisson_pmf_matrix(lam, R)  # n = 0..R-1
    n = np.arange(R)
    body = pmf @ v_next[R - n] + prices * (pmf @ n)
    tail = np.clip(1.0 - pmf.sum(axis=1), 0.0, 1.0)
    return body + tail * (prices * R + v_next[0])


def expected_sales(lam: np.ndarray, R: int) -> np.ndarray:
    """E[min(N, R)] for N ~ Poisson(lam)."""
    if R <= 0:
        return np.zeros(len(np.atleast_1d(lam)))
    pmf = poisson_pmf_matrix(np.atleast_1d(lam), R)
    n = np.arange(R)
    return pmf @ n + np.clip(1.0 - pmf.sum(axis=1), 0.0, 1.0) * R


def value_tables(price_sets: list[np.ndarray], lam_sets: list[np.ndarray], capacity: int) -> list[np.ndarray]:
    """Backward induction. Returns [V_0, ..., V_K], each of length capacity+1 (V_K = 0).

    price_sets[j] / lam_sets[j]: candidate prices and the model's demand means in window j.
    """
    K = len(price_sets)
    C = capacity
    V = [np.zeros(C + 1) for _ in range(K + 1)]
    R = np.arange(C + 1)[:, None]
    n = np.arange(C + 1)[None, :]
    below = n < R  # [R, n]
    for j in range(K - 1, -1, -1):
        p = np.asarray(price_sets[j], dtype=float)
        pmf = poisson_pmf_matrix(lam_sets[j], C + 1)  # [c, n]
        vn = V[j + 1]
        cont = np.where(below, vn[np.clip(R - n, 0, C)], 0.0)  # [R, n]
        body = pmf @ cont.T + p[:, None] * (pmf @ (below * n).T)  # [c, R]
        tail = np.clip(1.0 - pmf @ below.T, 0.0, 1.0)
        val = body + tail * (p[:, None] * R.T + vn[0])
        V[j] = val.max(axis=0)
    return V


def bid_price(v_next: np.ndarray, R: int) -> float:
    """Opportunity cost of selling one more room now."""
    if R <= 0:
        return float("inf")
    return float(v_next[R] - v_next[R - 1])


DemandFn = Callable[[object, np.ndarray], np.ndarray]


def night_value_tables(contexts: list, grid: np.ndarray, demand_fn: DemandFn, capacity: int) -> list[np.ndarray]:
    """Value tables for one night given per-window contexts and a demand model (price grid = static x grid)."""
    prices = [c.static * grid for c in contexts]
    lams = [np.maximum(demand_fn(c, p), 0.0) for c, p in zip(contexts, prices)]
    return value_tables(prices, lams, capacity)
