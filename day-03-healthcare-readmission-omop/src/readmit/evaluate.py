"""Discrimination, calibration, decision-curve and subgroup metrics for binary risk models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def logistic_irls(X: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None, iters: int = 50) -> np.ndarray:
    """Unpenalised logistic regression by Newton-Raphson (supports an offset). Returns coefficients."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta + off
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        grad = X.T @ (y - mu)
        hess = X.T @ (X * w[:, None])
        step = np.linalg.solve(hess + 1e-10 * np.eye(len(beta)), grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Calibration slope (logistic recalibration on logit p) and calibration-in-the-large (offset model)."""
    lp = _logit(p)
    slope = logistic_irls(np.column_stack([np.ones_like(lp), lp]), y)[1]
    citl = logistic_irls(np.ones((len(lp), 1)), y, offset=lp)[0]
    return float(slope), float(citl)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    slope, citl = calibration_slope_intercept(y, p)
    prev = y.mean()
    brier = brier_score_loss(y, p)
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "observed_rate": float(prev),
        "mean_predicted": float(p.mean()),
        "o_e_ratio": float(prev / p.mean()),
        "auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "auprc": float(average_precision_score(y, p)) if y.sum() > 0 else float("nan"),
        "brier": float(brier),
        "scaled_brier": float(1 - brier / (prev * (1 - prev))) if 0 < prev < 1 else float("nan"),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
        "calibration_slope": slope,
        "calibration_intercept": citl,
    }


def cluster_bootstrap(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_boot: int = 500, seed: int = 42
) -> dict[str, tuple[float, float]]:
    """95% percentile CIs for AUROC, Brier and calibration slope, resampling patients (not admissions)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_by_group = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    stats: dict[str, list[float]] = {"auroc": [], "brier": [], "calibration_slope": [], "calibration_intercept": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_group[g] for g in pick])
        yy, pp = y[idx], p[idx]
        if yy.min() == yy.max():
            continue
        s, c = calibration_slope_intercept(yy, pp)
        stats["auroc"].append(roc_auc_score(yy, pp))
        stats["brier"].append(brier_score_loss(yy, pp))
        stats["calibration_slope"].append(s)
        stats["calibration_intercept"].append(c)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in stats.items()}


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted risk with observed rate and Wilson 95% interval."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").agg(n=("y", "size"), events=("y", "sum"), mean_predicted=("p", "mean"))
    g["observed"] = g["events"] / g["n"]
    z = 1.96
    phat, n = g["observed"], g["n"]
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    g["ci_low"], g["ci_high"] = centre - half, centre + half
    return g.reset_index()


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Decision-curve net benefit: TP/n - FP/n * pt / (1 - pt) at each threshold pt."""
    y = np.asarray(y, dtype=int)
    n = len(y)
    out = []
    for pt in thresholds:
        treat = p >= pt
        tp = np.sum(treat & (y == 1))
        fp = np.sum(treat & (y == 0))
        out.append(tp / n - fp / n * pt / (1 - pt))
    return np.array(out)


def decision_curve(y: np.ndarray, preds: dict[str, np.ndarray], thresholds: np.ndarray) -> pd.DataFrame:
    prev = float(np.mean(y))
    df = pd.DataFrame({"threshold": thresholds})
    df["treat_all"] = prev - (1 - prev) * thresholds / (1 - thresholds)
    df["treat_none"] = 0.0
    for name, p in preds.items():
        df[name] = net_benefit(y, p, thresholds)
    return df


AGE_BANDS = [18, 50, 65, 75, 85, 200]
AGE_LABELS = ["18-49", "50-64", "65-74", "75-84", "85+"]


def subgroup_table(df: pd.DataFrame, y_col: str, p_col: str, min_events_for_slope: int = 20) -> pd.DataFrame:
    """Metrics by age band and sex. ``df`` needs age, sex, outcome and prediction columns."""
    d = df.copy()
    d["age_band"] = pd.cut(d["age"], AGE_BANDS, right=False, labels=AGE_LABELS).astype(str)
    rows = []
    for dim in ("age_band", "sex"):
        for level, g in d.groupby(dim, sort=True):
            y = g[y_col].to_numpy(int)
            p = g[p_col].to_numpy(float)
            row = {"dimension": dim, "level": level, "n": len(g), "events": int(y.sum())}
            row["observed_rate"] = float(y.mean())
            row["mean_predicted"] = float(p.mean())
            row["o_e_ratio"] = float(y.mean() / p.mean())
            row["auroc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
            row["brier"] = float(brier_score_loss(y, p))
            row["calibration_slope"] = (
                calibration_slope_intercept(y, p)[0] if y.sum() >= min_events_for_slope else float("nan")
            )
            rows.append(row)
    return pd.DataFrame(rows)


def kaplan_meier(time: np.ndarray, event: np.ndarray, horizon: int = 30) -> pd.DataFrame:
    """Kaplan-Meier survival (free of readmission) on integer days 0..horizon."""
    time = np.asarray(time, dtype=int)
    event = np.asarray(event, dtype=bool)
    surv, s = [], 1.0
    at_risk = len(time)
    for t in range(0, horizon + 1):
        d = int(np.sum((time == t) & event))
        c = int(np.sum((time == t) & ~event))
        if at_risk > 0:
            s *= 1 - d / at_risk
        surv.append({"day": t, "at_risk": at_risk, "events": d, "survival": s, "cumulative_incidence": 1 - s})
        at_risk -= d + c
    return pd.DataFrame(surv)
