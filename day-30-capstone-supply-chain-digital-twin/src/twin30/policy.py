"""Replenishment rules the retailer runs today.

These are business rules, not learned behaviour, so the SYNTHETIC world generator and the SimPy twin
both call the same functions. Everything that is uncertain (demand, lead times, outages, transit
delays) is modelled separately by each side.
"""

from __future__ import annotations

import numpy as np

from .config import Policy

# Lead times the planners quote when sizing DC base-stock (supplier contracts), not the realised ones.
QUOTED_LEAD_DAYS = np.array([21.0, 5.0])


def promo_in_window(promo: np.ndarray, t: int, cover: np.ndarray) -> np.ndarray:
    """promo: (T, C) 0/1 calendar; cover: (S,) days. Returns (S, C) bool: promo in [t+1, t+cover]."""
    T = promo.shape[0]
    out = np.zeros((cover.shape[0], promo.shape[1]), dtype=bool)
    for k in np.unique(cover):
        lo, hi = min(t + 1, T), min(t + 1 + int(k), T)
        flag = promo[lo:hi].any(axis=0) if hi > lo else np.zeros(promo.shape[1], dtype=bool)
        out[cover == k] = flag
    return out


def store_orders(
    on_hand: np.ndarray,
    in_transit: np.ndarray,
    sales_hist: np.ndarray,
    cover: np.ndarray,
    promo_soon: np.ndarray,
    policy: Policy,
) -> np.ndarray:
    """Order-up-to rule per store x category.

    on_hand, in_transit: (S, C); sales_hist: (W, S, C) trailing daily sales; cover: (S,) days of
    cover (transit + 1 review day); promo_soon: (S, C) bool.
    """
    fc = sales_hist.mean(axis=0)
    sd = sales_hist.std(axis=0) + 1e-9
    lift = np.where(promo_soon, policy.planned_promo_lift, 1.0)
    cov = cover[:, None].astype(float)
    target = cov * fc * lift + policy.store_z * sd * np.sqrt(cov)
    target = np.minimum(target, policy.max_cover_days * fc * lift)  # store space limit
    return np.maximum(0.0, np.ceil(target - on_hand - in_transit))


def ration(requests: np.ndarray, assign: np.ndarray, dc_stock: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Allocate DC stock to store requests.

    requests: (S, C); assign: (S,) DC index; dc_stock: (D, C); capacity: (D,) units/day.
    Shortages are shared pro rata, then the DC's daily throughput cap is applied pro rata.
    """
    ship = np.zeros_like(requests)
    for d in range(dc_stock.shape[0]):
        rows = assign == d
        if not rows.any():
            continue
        req = requests[rows]
        tot = req.sum(axis=0)
        frac = np.where(tot > dc_stock[d], dc_stock[d] / np.maximum(tot, 1e-9), 1.0)
        alloc = np.floor(req * frac[None, :])
        total = alloc.sum()
        if total > capacity[d]:
            alloc = np.floor(alloc * capacity[d] / total)
        ship[rows] = alloc
    return ship


def dc_orders(
    dc_on_hand: np.ndarray,
    dc_on_order: np.ndarray,
    outflow_hist: np.ndarray,
    policy: Policy,
) -> np.ndarray:
    """Weekly base-stock order per DC x category x supplier.

    dc_on_hand, dc_on_order: (D, C); outflow_hist: (W, D, C) trailing daily shipments to stores.
    Returns (D, C, 2) units for [offshore, nearshore].
    """
    share = np.array([1.0 - policy.nearshore_share, policy.nearshore_share])
    lead = float(share @ QUOTED_LEAD_DAYS)
    cover = lead + policy.dc_review_days
    mu = outflow_hist.mean(axis=0)
    sd = outflow_hist.std(axis=0)
    target = cover * mu + policy.dc_z * sd * np.sqrt(cover)
    need = np.maximum(0.0, target - dc_on_hand - dc_on_order)
    return np.floor(need[:, :, None] * share[None, None, :])
