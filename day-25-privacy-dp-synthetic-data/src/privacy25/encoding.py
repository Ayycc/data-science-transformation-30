"""Shared encodings: public-bin discretisation, model features and Gower-style distances."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORICAL, COLUMNS, NUMERIC_BOUNDS, TARGET

# Public bin edges (chosen from the questionnaire ranges, never from the private data). Right-open bins.
BINS: dict[str, list[float]] = {
    "age": [17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 91],
    "hours_per_week": [1, 20, 30, 35, 40, 41, 45, 50, 60, 100],
    "capital_gain": [0, 1, 2000, 4000, 7000, 10000, 20000, 99999, 100000],
}


def domain_sizes() -> dict[str, int]:
    return {c: (len(BINS[c]) - 1 if c in BINS else len(CATEGORICAL[c])) for c in COLUMNS}


def discretize(df: pd.DataFrame) -> np.ndarray:
    """Integer codes (n, d) in ``COLUMNS`` order; numerics go to public bins, categoricals to their level index."""
    out = np.empty((len(df), len(COLUMNS)), dtype=np.int64)
    for j, c in enumerate(COLUMNS):
        if c in BINS:
            edges = np.asarray(BINS[c], dtype=float)
            out[:, j] = np.clip(np.searchsorted(edges, df[c].to_numpy(dtype=float), side="right") - 1, 0, len(edges) - 2)
        else:
            levels = {v: i for i, v in enumerate(CATEGORICAL[c])}
            out[:, j] = df[c].map(levels).to_numpy()
    return out


def undiscretize(codes: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Back to the microdata schema; numerics are drawn uniformly (integers) inside their bin."""
    cols: dict[str, np.ndarray] = {}
    for j, c in enumerate(COLUMNS):
        k = codes[:, j]
        if c in BINS:
            edges = np.asarray(BINS[c], dtype=float)
            lo, hi = edges[k], edges[k + 1]
            v = np.floor(lo + rng.random(len(k)) * (hi - lo))
            lo_b, hi_b = NUMERIC_BOUNDS[c]
            cols[c] = np.clip(v, lo_b, hi_b).astype(int)
        else:
            cols[c] = np.asarray(CATEGORICAL[c], dtype=object)[k]
    return pd.DataFrame(cols)[COLUMNS]


def model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Features with pandas categoricals (fixed public levels) for HistGradientBoosting, and the 0/1 label."""
    X = df.drop(columns=[TARGET]).copy()
    for c, levels in CATEGORICAL.items():
        if c in X:
            X[c] = pd.Categorical(X[c], categories=levels)
    for c in NUMERIC_BOUNDS:
        X[c] = X[c].astype(float)
    y = (df[TARGET] == ">50K").to_numpy().astype(int)
    return X, y


def linear_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """One-hot + public-range-scaled numerics for linear models, and a *public* L2 bound on every row.

    Each numeric is scaled to [0, 1] with its public bounds (capital gain on a log scale) and each categorical
    block is one-hot (norm 1), so ||x||_2 <= sqrt(#columns) holds for every possible record: that is the
    ``data_norm`` diffprivlib needs without looking at the data.
    """
    parts = []
    for c, (lo, hi) in NUMERIC_BOUNDS.items():
        v = df[c].to_numpy(dtype=float)
        if c == "capital_gain":
            parts.append((np.log1p(v) / np.log1p(hi))[:, None])
        else:
            parts.append(((v - lo) / (hi - lo))[:, None])
    for c, levels in CATEGORICAL.items():
        if c == TARGET:
            continue
        idx = pd.Categorical(df[c], categories=levels).codes
        oh = np.zeros((len(df), len(levels)))
        ok = idx >= 0
        oh[np.where(ok)[0], idx[ok]] = 1.0
        parts.append(oh)
    X = np.hstack(parts)
    n_cols = len(NUMERIC_BOUNDS) + len(CATEGORICAL) - 1
    y = (df[TARGET] == ">50K").to_numpy().astype(int)
    return X, y, float(np.sqrt(n_cols))


def gower_matrix_inputs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Numeric block scaled by public range and categorical codes, for Gower distances."""
    num = np.column_stack([(df[c].to_numpy(dtype=float) - lo) / (hi - lo) for c, (lo, hi) in NUMERIC_BOUNDS.items()])
    cat = np.column_stack([pd.Categorical(df[c], categories=lv).codes for c, lv in CATEGORICAL.items()]).astype(np.int16)
    return num, cat


def nearest_gower(query: pd.DataFrame, reference: pd.DataFrame, chunk: int = 256) -> np.ndarray:
    """Gower distance from every query row to its nearest reference row (mean of per-column distances)."""
    qn, qc = gower_matrix_inputs(query)
    rn, rc = gower_matrix_inputs(reference)
    d_cols = qn.shape[1] + qc.shape[1]
    out = np.empty(len(query))
    for s in range(0, len(query), chunk):
        a_n, a_c = qn[s : s + chunk], qc[s : s + chunk]
        dist = np.abs(a_n[:, None, :] - rn[None, :, :]).sum(-1)
        dist += (a_c[:, None, :] != rc[None, :, :]).sum(-1)
        out[s : s + chunk] = dist.min(1) / d_cols
    return out
