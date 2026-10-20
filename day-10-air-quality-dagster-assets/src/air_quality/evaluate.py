"""Scores the published alerts and drift flags against the generator's ground truth (SYNTHETIC data only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .aqi import daily_station_aqi
from .config import ALERT_AQI, START_DATE


def _prf(pred: set, true: set) -> dict:
    tp = len(pred & true)
    p = tp / len(pred) if pred else float("nan")
    r = tp / len(true) if true else float("nan")
    f1 = 2 * p * r / (p + r) if pred and true and (p + r) > 0 else float("nan")
    return {
        "predicted": len(pred),
        "true": len(true),
        "true_positives": tp,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
    }


def _keys(df: pd.DataFrame) -> set:
    return set(zip(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"), df["station_id"], strict=True))


def truth_daily_aqi(data: Path) -> pd.DataFrame:
    truth = pd.read_parquet(Path(data) / "truth" / "truth_hourly.parquet")
    return daily_station_aqi(truth)


def naive_daily_aqi(data: Path) -> pd.DataFrame:
    """What the old cron chain computed: every landed reading, no cleaning, no calibration, no gating."""
    files = sorted((Path(data) / "landing").glob("*/batch_*.parquet"))
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    raw = raw.sort_values("landed_at").drop_duplicates(["station_id", "ts"], keep="last")
    raw[["pm25", "no2", "o3"]] = raw[["pm25", "no2", "o3"]].clip(lower=0)
    return daily_station_aqi(raw[["station_id", "ts", "pm25", "no2", "o3"]])


def alert_quality(conn, data: Path, published_dates: list[str]) -> dict:
    truth = truth_daily_aqi(data)
    true_alerts = _keys(truth[truth["aqi"] > ALERT_AQI])
    pipe = conn.execute("SELECT date, station_id FROM alerts").df()
    pipe_alerts = _keys(pipe) if len(pipe) else set()
    naive = naive_daily_aqi(data)
    naive_alerts = _keys(naive[naive["aqi"] > ALERT_AQI])
    pub = set(published_dates)
    true_pub = {k for k in true_alerts if k[0] in pub}
    out = {
        "truth_station_day_alerts": len(true_alerts),
        "pipeline_all_days": _prf(pipe_alerts, true_alerts),
        "naive_all_days": _prf(naive_alerts, true_alerts),
        "pipeline_published_days": _prf(pipe_alerts, true_pub),
        "naive_published_days": _prf({k for k in naive_alerts if k[0] in pub}, true_pub),
    }
    st = pd.read_parquet(Path(data) / "truth" / "stations_truth.parquet")
    drifters = set(st.loc[st["drift_start_day"] >= 0, "station_id"])
    out["naive_false_alerts_from_drifting_stations"] = len({k for k in naive_alerts - true_alerts if k[1] in drifters})
    out["pipeline_false_alerts_from_drifting_stations"] = len(
        {k for k in pipe_alerts - true_alerts if k[1] in drifters}
    )
    out["naive_false_alerts_on_unit_bug_day"] = len({k for k in naive_alerts - true_alerts if k[0] == "2026-09-24"})
    by_cat = truth[truth["aqi"] > ALERT_AQI]["category"].value_counts().to_dict()
    out["truth_alerts_by_category"] = {k: int(v) for k, v in by_cat.items()}
    return out


def drift_quality(conn, data: Path) -> dict:
    st = pd.read_parquet(Path(data) / "truth" / "stations_truth.parquet")
    drift = conn.execute("SELECT date, station_id, drift_rate, drift_suspect FROM sensor_drift").df()
    drift["date"] = pd.to_datetime(drift["date"])
    last = drift["date"].max()
    flagged_last = set(drift.loc[(drift["date"] == last) & drift["drift_suspect"], "station_id"])
    truth = set(st.loc[st["drift_start_day"] >= 0, "station_id"])
    ever = set(drift.loc[drift["drift_suspect"], "station_id"])
    false_station_days = int((drift["drift_suspect"] & ~drift["station_id"].isin(truth)).sum())
    delays = []
    t0 = pd.Timestamp(START_DATE)
    for _, r in st[st["drift_start_day"] >= 0].iterrows():
        f = drift[(drift["station_id"] == r["station_id"]) & drift["drift_suspect"]]
        if len(f):
            delays.append(
                float((f["date"].min() - (t0 + pd.Timedelta(days=int(r["drift_start_day"])))) / pd.Timedelta(days=1))
            )
    est = drift[(drift["date"] == last) & drift["station_id"].isin(truth)].merge(st, on="station_id")
    rate_err = (est["drift_rate_x"] - est["drift_rate_y"]).abs()
    return {
        "drifting_stations_true": len(truth),
        "on_last_day": _prf(flagged_last, truth),
        "ever_flagged": _prf(ever, truth),
        "false_suspect_station_days": false_station_days,
        "detection_delay_days_median": float(np.median(delays)) if delays else None,
        "detection_delay_days_max": float(np.max(delays)) if delays else None,
        "rate_abs_error_median_ug_m3_per_day": round(float(rate_err.median()), 3) if len(est) else None,
    }
