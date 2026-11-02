"""Optional public sanity check: Numenta Anomaly Benchmark (NAB, AGPL-3.0 data, github.com/numenta/NAB).

NAB has real infrastructure metrics with labelled anomaly windows but no service topology, so it is used only to
sanity-check the detectors (window recall, false alarms). Root-cause analysis runs on the SYNTHETIC platform.
If GitHub is unreachable the check is skipped and the reason is recorded."""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest

from .detect import episodes, persistence

BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
FILES = [
    "realKnownCause/ec2_request_latency_system_failure.csv",
    "realKnownCause/cpu_utilization_asg_misconfiguration.csv",
    "realKnownCause/machine_temperature_system_failure.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
    "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv",
    "realAWSCloudwatch/elb_request_count_8c0756.csv",
]
PROBATION = 0.15  # NAB convention: first 15% of each series is the learning period


def _get(url: str, timeout: float = 20.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (fixed https URL)
        return r.read()


def fetch(cache: Path) -> tuple[dict[str, pd.DataFrame], dict[str, list]]:
    cache.mkdir(parents=True, exist_ok=True)
    lab_path = cache / "combined_windows.json"
    if not lab_path.exists():
        lab_path.write_bytes(_get(f"{BASE}/labels/combined_windows.json"))
    labels = json.loads(lab_path.read_text())
    series = {}
    for f in FILES:
        p = cache / f.replace("/", "__")
        if not p.exists():
            p.write_bytes(_get(f"{BASE}/data/{f}"))
        series[f] = parse_series(p.read_bytes())
    return series, {f: labels.get(f, []) for f in FILES}


def parse_series(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[["timestamp", "value"]].sort_values("timestamp").reset_index(drop=True)


def window_mask(ts: pd.Series, windows: list) -> np.ndarray:
    m = np.zeros(len(ts), dtype=bool)
    for a, b in windows:
        m |= ((ts >= pd.Timestamp(a)) & (ts <= pd.Timestamp(b))).to_numpy()
    return m


def evaluate_series(df: pd.DataFrame, windows: list, seed: int) -> dict:
    x = df["value"].to_numpy(float)
    n = len(x)
    ntr = int(PROBATION * n)
    med = np.median(x[:ntr])
    mad = 1.4826 * np.median(np.abs(x[:ntr] - med)) + 1e-9
    z = (x - med) / mad
    dz = np.r_[0, np.diff(z)]
    rm = pd.Series(z).rolling(5, min_periods=1).mean().to_numpy()
    X = np.column_stack([z, rm, np.abs(dz)])
    out = {"n_points": n, "n_windows": len(windows)}
    wins = [window_mask(df["timestamp"], [w]) for w in windows]
    inside = window_mask(df["timestamp"], windows)
    dets = {"univariate_threshold": persistence(np.abs(z) > 5, 2)}
    for name, model in (("iforest", IForest(n_estimators=150, random_state=seed)), ("ecod", ECOD())):
        model.fit(X[:ntr])
        thr = np.quantile(model.decision_scores_, 0.995)
        dets[name] = persistence(model.decision_function(X) > thr, 2)
    for name, flags in dets.items():
        flags = flags.copy()
        flags[:ntr] = False
        ep = episodes(flags[:, None], flags[:, None].astype(float), ["s"], gap=3)
        fp = sum(1 for e in ep.itertuples() if not inside[e.start : e.end + 1].any())
        hit = sum(1 for w in wins if (flags & w).any())
        out[name] = dict(windows_hit=hit, false_alarm_episodes=fp, false_alarms_per_1k_points=round(1000 * fp / n, 3))
    return out


def run_public_check(cache: Path, seed: int) -> dict:
    try:
        series, labels = fetch(cache)
    except Exception as exc:  # proxy 403, offline, DNS ...
        return {"path_used": "skipped", "reason": f"{type(exc).__name__}: {exc}"[:200]}
    per = {f: evaluate_series(series[f], labels[f], seed) for f in FILES}
    tot = {}
    for det in ("univariate_threshold", "iforest", "ecod"):
        hit = sum(v[det]["windows_hit"] for v in per.values())
        nw = sum(v["n_windows"] for v in per.values())
        fp = sum(v[det]["false_alarm_episodes"] for v in per.values())
        npts = sum(v["n_points"] for v in per.values())
        tot[det] = dict(
            window_recall=round(hit / nw, 4),
            windows_hit=hit,
            n_windows=nw,
            false_alarm_episodes=fp,
            false_alarms_per_1k_points=round(1000 * fp / npts, 3),
        )
    return {"path_used": "public_nab", "source": BASE, "files": FILES, "totals": tot, "per_file": per}
