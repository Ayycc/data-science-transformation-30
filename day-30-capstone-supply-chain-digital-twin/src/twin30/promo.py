"""Promotion calendars: the merchandisers' current rule, a posterior-targeted rule, and none."""

from __future__ import annotations

import numpy as np

WEEKS = 52


def weekly_to_daily(weeks: np.ndarray) -> np.ndarray:
    """(52, C) 0/1 weekly flags -> (364, C) daily flags."""
    return np.repeat(weeks, 7, axis=0)


def post_promo(daily: np.ndarray, prev_last_week: np.ndarray | None = None) -> np.ndarray:
    """Flag the week after a promo week (if that week is not itself a promo week)."""
    weeks = daily[::7]
    prev = np.zeros(weeks.shape[1]) if prev_last_week is None else prev_last_week
    shifted = np.vstack([prev[None, :], weeks[:-1]])
    post_w = ((shifted > 0) & (weeks == 0)).astype(float)
    return weekly_to_daily(post_w)


def current_rule(season_index: np.ndarray, n_weeks: list[int], rng: np.random.Generator) -> np.ndarray:
    """The merchandisers' rule: promote in high-season weeks, never two weeks in a row.

    season_index: (52, C) relative weekly demand index (1 = average). Selection probability is
    proportional to index**3, which is why naive promo-vs-no-promo comparisons are confounded by
    seasonality. Returns (52, C) 0/1.
    """
    out = np.zeros((WEEKS, season_index.shape[1]))
    for c, k in enumerate(n_weeks):
        w = np.clip(season_index[:, c], 1e-6, None) ** 3
        chosen: list[int] = []
        avail = np.ones(WEEKS, dtype=bool)
        while len(chosen) < k and avail.any():
            p = np.where(avail, w, 0.0)
            j = int(rng.choice(WEEKS, p=p / p.sum()))
            chosen.append(j)
            avail[max(0, j - 1) : j + 2] = False
        out[chosen, c] = 1.0
    return out


def targeted_rule(week_value: np.ndarray, n_weeks: list[int]) -> np.ndarray:
    """Pick, per category, the non-adjacent weeks with the highest expected incremental margin.

    week_value: (52, C) posterior expected incremental gross margin of a promo in that week
    (net of the discount and the post-promo dip). Weeks with negative value are never promoted,
    so a category may get fewer promo weeks than its budget.
    """
    out = np.zeros_like(week_value)
    for c, k in enumerate(n_weeks):
        order = np.argsort(-week_value[:, c])
        avail = np.ones(WEEKS, dtype=bool)
        n = 0
        for j in order:
            if n >= k or week_value[j, c] <= 0:
                break
            if avail[j]:
                out[j, c] = 1.0
                avail[max(0, j - 1) : j + 2] = False
                n += 1
    return out
