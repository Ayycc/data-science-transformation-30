"""Inventory policies: service-level (s, S), newsvendor for seasonal items, and the planners' gut-feel heuristic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .forecasting import lead_time_demand, quantile


def round_to_pack(q: np.ndarray, pack: np.ndarray, moq: np.ndarray) -> np.ndarray:
    """Round a positive order up to whole packs and to at least the MOQ; zero stays zero."""
    q = np.asarray(q, float)
    packs = np.ceil(np.maximum(q, 0) / pack) * pack
    return np.where(q > 0, np.maximum(packs, moq), 0).astype(np.int64)


def costs_per_item(m: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-unit holding cost per week, per-unit shortage cost and ordering cost, all from config."""
    c = cfg["costs"]
    out = pd.DataFrame(index=m.index)
    out["h_week"] = m["unit_cost"] * c["holding_rate_annual"] / 52.0
    out["shortage"] = (m["price"] - m["unit_cost"]) + c["shortage_goodwill_frac_price"] * m["price"]
    out["K"] = float(c["ordering_cost_per_line"])
    return out


def optimised_sS(fc: pd.DataFrame, m: pd.DataFrame, seg: pd.DataFrame, cfg: dict, lt_cv_mult: float = 1.0) -> pd.DataFrame:
    """(s, S) per item: s = CSL-quantile of lead-time demand, S = s + max(EOQ, MOQ) rounded to packs."""
    m = m.loc[fc.index]
    R = float(cfg["policy"]["review_period_weeks"])
    lt_mean = m["lt_mean_weeks"].to_numpy()
    lt_sd = lt_mean * m["lt_cv"].to_numpy() * lt_cv_mult
    mu, sigma = fc["mu"].to_numpy(), fc["sigma"].to_numpy()
    mean_p, sd_p = lead_time_demand(mu, sigma, lt_mean, lt_sd, R)
    targets = cfg["service_targets"]
    csl = seg.loc[fc.index, "segment"].map(targets).fillna(min(targets.values())).to_numpy()
    s = np.ceil(quantile(mean_p, sd_p, csl, fc["dist"].to_numpy()))
    k = costs_per_item(m, cfg)
    annual = mu * 52
    eoq = np.sqrt(2 * k["K"].to_numpy() * annual / np.maximum(k["h_week"].to_numpy() * 52, 1e-9))
    pack, moq = m["pack_size"].to_numpy(), m["moq"].to_numpy()
    Q = round_to_pack(np.maximum(eoq, 1.0), pack, moq)
    s = np.where(mu > 0, s, 0)
    S = np.where(mu > 0, s + Q, 0)
    return pd.DataFrame(
        {"s": s.astype(int), "S": S.astype(int), "csl_target": csl, "ltd_mean": mean_p, "ltd_sd": sd_p, "eoq": eoq},
        index=fc.index,
    )


def heuristic_sS(history: pd.DataFrame, m: pd.DataFrame, cfg: dict, items: pd.Index) -> pd.DataFrame:
    """Gut-feel rule: s = 1.5 x recent 13-week average x mean lead time; S = s + 4 weeks of average demand."""
    h = cfg["policy"]["heuristic"]
    n = int(history["week"].max()) + 1
    recent = history[(history["week"] >= n - h["avg_window_weeks"]) & history["item_id"].isin(items)]
    avg = recent.groupby("item_id")["demand"].mean().reindex(items).fillna(0.0)
    lt = m.loc[items, "lt_mean_weeks"]
    s = np.ceil(h["reorder_cover_factor"] * avg * lt)
    S = np.ceil(s + h["order_up_to_extra_weeks"] * avg)
    return pd.DataFrame({"s": s.astype(int), "S": S.astype(int), "avg": avg}, index=items)


def critical_ratio(price: np.ndarray, cost: np.ndarray, salvage: np.ndarray, goodwill: np.ndarray) -> np.ndarray:
    cu = price - cost + goodwill
    co = cost - salvage
    return cu / (cu + co)


def season_totals(history: pd.DataFrame, items: pd.Index, season_mask: np.ndarray) -> pd.DataFrame:
    h = history[history["item_id"].isin(items)].copy()
    h = h[season_mask[h["week"].to_numpy()]]
    h["year"] = h["week"] // 52
    return h.pivot_table(index="item_id", columns="year", values="demand", aggfunc="sum").reindex(items).fillna(0)


def newsvendor(history: pd.DataFrame, m: pd.DataFrame, items: pd.Index, season_mask: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Single pre-season buy at the critical-ratio quantile of a lognormal season-total forecast.

    The season-to-season log spread is pooled across all seasonal items (two seasons per item is too few alone).
    """
    tot = season_totals(history, items, season_mask)
    last = tot.iloc[:, -1].to_numpy()
    mean_tot = tot.mean(axis=1).to_numpy()
    if tot.shape[1] >= 2:
        diffs = np.log(np.maximum(tot.iloc[:, -1], 1)) - np.log(np.maximum(tot.iloc[:, -2], 1))
        sig = float(np.std(diffs, ddof=1) / np.sqrt(2)) if len(diffs) > 1 else 0.3
    else:
        sig = 0.3
    sig = max(sig, 0.05)
    mm = m.loc[items]
    c = cfg["costs"]
    salvage = mm["unit_cost"].to_numpy() * c["seasonal_salvage_frac_cost"]
    cr = critical_ratio(
        mm["price"].to_numpy(), mm["unit_cost"].to_numpy(), salvage, c["shortage_goodwill_frac_price"] * mm["price"].to_numpy()
    )
    mu_log = np.log(np.maximum(mean_tot, 1e-6)) - sig**2 / 2
    q_star = np.where(mean_tot > 0, np.exp(mu_log + sig * stats.norm.ppf(cr)), 0.0)
    pack, moq = mm["pack_size"].to_numpy(), mm["moq"].to_numpy()
    mult = cfg["policy"]["heuristic"]["seasonal_last_season_multiplier"]
    return pd.DataFrame(
        {
            "critical_ratio": cr,
            "season_mean": mean_tot,
            "sigma_log": sig,
            "q_newsvendor": round_to_pack(q_star, pack, moq),
            "q_heuristic": round_to_pack(last * mult, pack, moq),
        },
        index=items,
    )
