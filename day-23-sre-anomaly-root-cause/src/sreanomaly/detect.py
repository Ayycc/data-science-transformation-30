"""Per-service anomaly detection: seasonal robust z-scores, univariate threshold alerts (the status quo),
and multivariate detectors (PyOD Isolation Forest and ECOD) fitted on an incident-free training window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest

BIN_MIN = 15
MIN_PER_DAY = 1440


def transform(values: np.ndarray) -> np.ndarray:
    """Variance-stabilising transforms per metric: log latency, sqrt error rate, raw CPU, log1p queue."""
    out = np.empty_like(values, dtype=np.float64)
    out[..., 0] = np.log(np.maximum(values[..., 0], 1e-3))
    out[..., 1] = np.sqrt(np.maximum(values[..., 1], 0))
    out[..., 2] = values[..., 2]
    out[..., 3] = np.log1p(np.maximum(values[..., 3], 0))
    return out


def seasonal_zscores(values: np.ndarray, train_minutes: int) -> np.ndarray:
    """Robust z-score against a 15-minute-of-day median/MAD profile learnt on the training window.

    values: (T, S, M). Returns (T, S, M)."""
    x = transform(values)
    T = x.shape[0]
    bins = (np.arange(T) % MIN_PER_DAY) // BIN_MIN
    nb = MIN_PER_DAY // BIN_MIN
    med = np.zeros((nb,) + x.shape[1:])
    mad = np.zeros_like(med)
    tr = x[:train_minutes]
    tb = bins[:train_minutes]
    for b in range(nb):
        blk = tr[tb == b]
        med[b] = np.median(blk, axis=0)
        mad[b] = np.median(np.abs(blk - med[b]), axis=0)
    # smooth the profile circularly (3 bins) so a single noisy bin does not create false alerts
    med = (np.roll(med, 1, 0) + med + np.roll(med, -1, 0)) / 3
    global_mad = np.median(mad, axis=0, keepdims=True)
    scale = 1.4826 * np.maximum(mad, 0.5 * global_mad) + 1e-6
    return (x - med[bins]) / scale[bins]


def features(z: np.ndarray) -> np.ndarray:
    """Per-minute features per service: current z (4) + trailing 5-minute mean z (4). Shape (T, S, 8)."""
    c = np.cumsum(np.concatenate([np.zeros((1,) + z.shape[1:]), z], axis=0), axis=0)
    T = z.shape[0]
    lo = np.maximum(0, np.arange(T) - 4)
    rm = (c[np.arange(T) + 1] - c[lo]) / (np.arange(T) + 1 - lo)[:, None, None]
    return np.concatenate([z, rm], axis=2)


@dataclass
class DetectorOutput:
    name: str
    scores: np.ndarray  # (T, S) normalised: score / threshold (>1 = anomalous)
    flags: np.ndarray  # (T, S) bool after persistence filter


def persistence(raw: np.ndarray, k: int = 2) -> np.ndarray:
    """Flag minute t only if the last k minutes all exceed the threshold."""
    out = raw.copy()
    for lag in range(1, k):
        out[lag:] &= raw[:-lag]
        out[:lag] = False
    return out


def univariate(z: np.ndarray, thr: float = 5.0, k: int = 2) -> DetectorOutput:
    """Status-quo threshold alerting: any metric with |z| > thr. Upward deviations only (SRE alerts fire on high)."""
    s = np.max(z, axis=2) / thr
    return DetectorOutput("univariate_threshold", s, persistence(s > 1, k))


def multivariate(feats: np.ndarray, train_minutes: int, kind: str, seed: int, quantile: float = 0.999, k: int = 2) -> DetectorOutput:
    """Fit one PyOD model per service on the training window; threshold = training-score quantile."""
    T, S, _ = feats.shape
    scores = np.zeros((T, S))
    for j in range(S):
        X = feats[:, j, :]
        model = IForest(n_estimators=150, max_samples=512, random_state=seed + j) if kind == "iforest" else ECOD()
        model.fit(X[:train_minutes])
        tr_scores = model.decision_scores_
        thr = np.quantile(tr_scores, quantile)
        sc = model.decision_function(X)
        lo = np.median(tr_scores)
        scores[:, j] = (sc - lo) / (thr - lo + 1e-9)
    return DetectorOutput(kind, scores, persistence(scores > 1, k))


def episodes(flags: np.ndarray, scores: np.ndarray, services: list[str], gap: int = 5) -> pd.DataFrame:
    """Turn per-minute flags into alert episodes per service (merge runs separated by <= gap minutes)."""
    rows = []
    T, S = flags.shape
    for j in range(S):
        on = np.flatnonzero(flags[:, j])
        if on.size == 0:
            continue
        splits = np.flatnonzero(np.diff(on) > gap + 1)
        starts = np.r_[on[0], on[splits + 1]]
        ends = np.r_[on[splits], on[-1]]
        for a, b in zip(starts, ends):
            rows.append(dict(service=services[j], start=int(a), end=int(b), peak=float(scores[a : b + 1, j].max())))
    df = pd.DataFrame(rows, columns=["service", "start", "end", "peak"])
    return df.sort_values(["start", "service"]).reset_index(drop=True)


def raw_metric_alerts(z: np.ndarray, services: list[str], metrics: list[str], thr: float = 5.0, k: int = 2, gap: int = 5) -> pd.DataFrame:
    """Raw alert notifications as a per-(service, metric) threshold rule would page them."""
    out = []
    for m, name in enumerate(metrics):
        s = z[:, :, m] / thr
        ep = episodes(persistence(s > 1, k), s, services, gap)
        ep["metric"] = name
        out.append(ep)
    return pd.concat(out, ignore_index=True).sort_values(["start", "service"]).reset_index(drop=True)
