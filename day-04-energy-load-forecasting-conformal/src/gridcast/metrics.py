"""Forecast accuracy and calibration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mase_scale(y_train: np.ndarray, m: int = 168) -> float:
    """In-sample MAE of the seasonal-naive forecast with period m (the MASE denominator)."""
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        raise ValueError(f"need more than {m} training points, got {len(y_train)}")
    return float(np.mean(np.abs(y_train[m:] - y_train[:-m])))


def mase(y: np.ndarray, yhat: np.ndarray, scale: float) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))) / scale)


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Symmetric MAPE in percent, 0-200 scale."""
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    denom = np.abs(y) + np.abs(yhat)
    return float(np.mean(np.where(denom == 0, 0.0, 2.0 * np.abs(y - yhat) / denom)) * 100)


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss of predictions q_pred for quantile level q."""
    d = np.asarray(y, dtype=float) - np.asarray(q_pred, dtype=float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """Mean interval (Winkler) score: width plus 2/alpha times the miss distance."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    score = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return float(np.mean(score))


def summarise(pred: pd.DataFrame, scales: pd.DataFrame, alpha: float = 0.10) -> pd.DataFrame:
    """Per model: MASE, sMAPE, scaled pinball, coverage, width.

    pred: columns model, region, fold, ds, horizon, y, q05, q50, q95.
    scales: columns region, fold, scale (MASE denominator for that region and fold).
    MASE and scaled pinball are computed per region and fold, then averaged with equal weight.
    """
    p = pred.merge(scales, on=["region", "fold"], how="left", validate="many_to_one")
    rows = []
    for model, g in p.groupby("model", sort=False):
        per = []
        for (_, _), gg in g.groupby(["region", "fold"]):
            s = gg["scale"].iloc[0]
            pin = np.mean([pinball(gg["y"], gg[c], q) for c, q in (("q05", 0.05), ("q50", 0.5), ("q95", 0.95))])
            per.append((mase(gg["y"], gg["q50"], s), pin / s))
        per = np.array(per)
        rows.append(
            {
                "model": model,
                "mase": float(per[:, 0].mean()),
                "smape_pct": smape(g["y"], g["q50"]),
                "scaled_pinball": float(per[:, 1].mean()),
                "pinball_mw": float(
                    np.mean([pinball(g["y"], g[c], q) for c, q in (("q05", 0.05), ("q50", 0.5), ("q95", 0.95))])
                ),
                "mae_mw": float(np.mean(np.abs(g["y"] - g["q50"]))),
                "coverage90": coverage(g["y"], g["q05"], g["q95"]),
                "mean_width_pct_of_load": float(np.mean((g["q95"] - g["q05"]) / g["y"]) * 100),
                "winkler_mw": winkler(g["y"], g["q05"], g["q95"], alpha),
                "n_points": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def coverage_table(pred: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Empirical coverage of [q05, q95] grouped by the given columns."""
    hit = (pred["y"] >= pred["q05"]) & (pred["y"] <= pred["q95"])
    return pred.assign(hit=hit).groupby(by, as_index=False).agg(coverage=("hit", "mean"), n=("hit", "size"))
