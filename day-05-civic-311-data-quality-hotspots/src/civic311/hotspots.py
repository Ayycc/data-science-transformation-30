"""H3 hexagonal aggregation of resolution times and Getis-Ord Gi* hot/cold spots."""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from scipy import stats

from .config import FDR_Q, H3_RES, MATERIAL_EXCESS, MIN_CELL_N
from .geo import latlng_to_cells


def resolution_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """Completed, contract-passing requests with resolution hours and the category-adjusted log excess."""
    c = clean[clean.status == "Completed"].copy()
    c["res_hours"] = (c.closed_ts - c.created_ts).dt.total_seconds() / 3600
    c = c[c.res_hours > 0]
    lr = np.log(c.res_hours)
    c["log_excess"] = lr - lr.groupby(c.category).transform("mean")
    c["h3"] = latlng_to_cells(c.latitude.to_numpy(), c.longitude.to_numpy(), H3_RES)
    return c


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def getis_ord_gi_star(cells: list[str], x: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Gi* z-scores with binary weights over the k-ring (self included) among the observed cells.

    Returns (z, local_mean) where local_mean is the mean of x over each cell's neighbourhood window.
    """
    x = np.asarray(x, float)
    n = len(x)
    pos = {c: i for i, c in enumerate(cells)}
    xbar = x.mean()
    s = np.sqrt((x**2).mean() - xbar**2)
    z = np.empty(n)
    local = np.empty(n)
    for i, c in enumerate(cells):
        nb = [pos[j] for j in h3.grid_disk(c, k) if j in pos]
        w = len(nb)
        sx = x[nb].sum()
        denom = s * np.sqrt((n * w - w**2) / (n - 1))
        z[i] = (sx - xbar * w) / denom if denom > 0 else 0.0
        local[i] = sx / w
    return z, local


def hotspot_table(res: pd.DataFrame) -> pd.DataFrame:
    """Per H3 cell: volume, median hours, mean log excess, naive z, Gi* z, q-values and the hot/cold label."""
    g = res.groupby("h3").agg(
        n=("log_excess", "size"),
        mean_log_excess=("log_excess", "mean"),
        median_hours=("res_hours", "median"),
    )
    g = g[g.n >= MIN_CELL_N].copy()
    xbar = g.mean_log_excess.mean()
    resid_sd = res.log_excess.std()
    # naive per-cell z-score (no spatial pooling)
    g["z_naive"] = (g.mean_log_excess - xbar) / (resid_sd / np.sqrt(g.n))
    g["q_naive"] = bh_fdr(2 * stats.norm.sf(np.abs(g.z_naive)))
    z, local = getis_ord_gi_star(list(g.index), g.mean_log_excess.to_numpy())
    g["gi_z"] = z
    g["gi_q"] = bh_fdr(2 * stats.norm.sf(np.abs(z)))
    g["local_excess_pct"] = 100 * (np.exp(local - xbar) - 1)
    g["cell_excess_pct"] = 100 * (np.exp(g.mean_log_excess - xbar) - 1)
    hot_thr, cold_thr = 100 * MATERIAL_EXCESS, 100 * (1 / (1 + MATERIAL_EXCESS) - 1)
    g["gi_class"] = np.select(
        [
            (g.gi_q < FDR_Q) & (g.gi_z > 0) & (g.local_excess_pct >= hot_thr),
            (g.gi_q < FDR_Q) & (g.gi_z < 0) & (g.local_excess_pct <= cold_thr),
            g.gi_q < FDR_Q,
        ],
        ["hot", "cold", "significant_not_material"],
        "not_significant",
    )
    g["naive_class"] = np.select(
        [
            (g.q_naive < FDR_Q) & (g.z_naive > 0) & (g.cell_excess_pct >= hot_thr),
            (g.q_naive < FDR_Q) & (g.z_naive < 0) & (g.cell_excess_pct <= cold_thr),
        ],
        ["hot", "cold"],
        "none",
    )
    lat_lng = [h3.cell_to_latlng(c) for c in g.index]
    g["lat"] = [a for a, _ in lat_lng]
    g["lon"] = [b for _, b in lat_lng]
    return g.reset_index()


def score_against_truth(cells: pd.DataFrame, res_truth: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare detected hot/cold cells with the generator's true multiplier per cell (SYNTHETIC ground truth).

    res_truth: rows of `resolution_frame` joined with truth (needs h3 and true_multiplier). A cell is truly slow
    when its mean log true multiplier exceeds the scored-cell average by log(1 + MATERIAL_EXCESS).
    """
    t = res_truth.groupby("h3").true_multiplier.apply(lambda m: np.log(m).mean())
    c = cells.set_index("h3").join(t.rename("true_log_mult"))
    c["true_excess_pct"] = 100 * (np.exp(c.true_log_mult - c.true_log_mult.mean()) - 1)
    hot_thr, cold_thr = 100 * MATERIAL_EXCESS, 100 * (1 / (1 + MATERIAL_EXCESS) - 1)
    c["truth_class"] = np.select([c.true_excess_pct >= hot_thr, c.true_excess_pct <= cold_thr], ["hot", "cold"], "none")
    ring1 = {}
    hot_true = set(c.index[c.truth_class == "hot"])
    cold_true = set(c.index[c.truth_class == "cold"])
    for cell in c.index:
        disk = set(h3.grid_disk(cell, 1))
        ring1[cell] = (bool(disk & hot_true), bool(disk & cold_true))
    out = {}
    for method, col in (("gi_star", "gi_class"), ("naive_z", "naive_class")):
        for kind, idx in (("hot", 0), ("cold", 1)):
            pred = c[col] == kind
            true = c.truth_class == kind
            tp = int((pred & true).sum())
            near = int(sum(ring1[k][idx] for k in c.index[pred]))
            out[f"{method}_{kind}"] = {
                "flagged": int(pred.sum()), "true_cells": int(true.sum()), "true_positive": tp,
                "recall": round(tp / max(int(true.sum()), 1), 4),
                "precision": round(tp / max(int(pred.sum()), 1), 4),
                "precision_within_1_ring": round(near / max(int(pred.sum()), 1), 4),
            }  # fmt: skip
    return c.reset_index(), out
