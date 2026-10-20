"""Pure pandas transforms behind the assets: cleaning, station health and drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DRIFT_MIN_DAYS,
    DRIFT_MIN_SLOPE,
    DRIFT_MIN_TSTAT,
    NEGATIVE_NOISE_FLOOR,
    PHYSICAL_BOUNDS,
    POLLUTANTS,
)

FLATLINE_MIN_HOURS = 4
SPIKE_FACTOR = 4.0
SPIKE_FLOOR = {"pm25": 20.0, "no2": 30.0, "o3": 40.0}


def deduplicate(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep the latest delivery of each (station_id, ts)."""
    df = raw.sort_values(["station_id", "ts", "landed_at"], kind="stable")
    out = df.drop_duplicates(["station_id", "ts"], keep="last").reset_index(drop=True)
    return out, len(raw) - len(out)


def _flatline_mask(df: pd.DataFrame, col: str) -> pd.Series:
    """True where a value is part of a run of >= FLATLINE_MIN_HOURS identical consecutive hourly values."""
    s = df[col]
    prev_station = df["station_id"].shift()
    prev_ts = df["ts"].shift()
    same = (s == s.shift()) & (df["station_id"] == prev_station) & ((df["ts"] - prev_ts) == pd.Timedelta(hours=1))
    run_id = (~same).cumsum()
    run_len = run_id.map(run_id.value_counts())
    return (run_len >= FLATLINE_MIN_HOURS) & s.notna()


def clean_readings(raw: pd.DataFrame, stations: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deduplicate, flag and null out invalid values, apply known calibration offsets.

    Returns the cleaned frame (invalid values are NaN, plus per-row flag columns) and summary counts.
    """
    df, n_dup = deduplicate(raw)
    df = df.sort_values(["station_id", "ts"]).reset_index(drop=True)
    stats: dict[str, int] = {"rows_in": int(len(raw)), "duplicates_removed": int(n_dup)}
    n_vals = 0
    for p in POLLUTANTS:
        lo, hi = PHYSICAL_BOUNDS[p]
        v = df[p].astype(float)
        n_vals += int(v.notna().sum())
        out_of_range = (v < NEGATIVE_NOISE_FLOOR) | (v > hi)
        v = v.where(~out_of_range)
        v = v.clip(lower=lo)
        flat = _flatline_mask(df.assign(**{p: v}), p)
        med = v.groupby(df["station_id"]).transform(lambda x: x.rolling(5, center=True, min_periods=3).median())
        spike = (v > SPIKE_FACTOR * med + SPIKE_FLOOR[p]) & ~flat
        v = v.where(~(flat | spike))
        df[p] = v
        df[f"{p}_flag"] = np.select([out_of_range, flat, spike], ["out_of_range", "flatline", "spike"], default="")
        stats[f"{p}_out_of_range"] = int(out_of_range.sum())
        stats[f"{p}_flatline"] = int(flat.sum())
        stats[f"{p}_spike"] = int(spike.sum())
    stats["raw_values"] = n_vals
    stats["out_of_range_values"] = sum(stats[f"{p}_out_of_range"] for p in POLLUTANTS)

    cal = stations.set_index("station_id")
    df["pm25"] = df["pm25"] - df["station_id"].map(cal["cal_offset_pm25"]).fillna(0.0)
    df["no2"] = df["no2"] - df["station_id"].map(cal["cal_offset_no2"]).fillna(0.0)
    for p in ("pm25", "no2"):
        df[p] = df[p].clip(lower=0.0)
    df["region"] = df["station_id"].map(cal["region"])
    stats["rows_out"] = int(len(df))
    return df, stats


def station_health(cleaned: pd.DataFrame, stations: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Per station uptime and fault counts for one day; every registry station gets a row."""
    g = cleaned.groupby("station_id")
    out = pd.DataFrame(index=stations["station_id"])
    out["hours_reported"] = g.size()
    for p in POLLUTANTS:
        out[f"{p}_valid_hours"] = g[p].count()
    flags = cleaned[[f"{p}_flag" for p in POLLUTANTS]].ne("").any(axis=1)
    out["flagged_hours"] = flags.groupby(cleaned["station_id"]).sum()
    out = out.fillna(0).astype(int)
    out["uptime"] = (out["pm25_valid_hours"] / 24.0).round(4)
    out["status"] = np.select(
        [out["hours_reported"] == 0, (out["pm25_valid_hours"] < 18) | (out["flagged_hours"] > 6)],
        ["offline", "degraded"],
        default="ok",
    )
    out = out.reset_index().rename(columns={"index": "station_id"})
    out.insert(0, "date", date)
    return out


def detect_drift(window: pd.DataFrame, stations: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Flag PM2.5 sensors whose bias against their regional median grows linearly over the window.

    For each station fit  station_pm25 = a + b * regional_median + g * day  on cleaned hourly values of the
    trailing window; ``g`` is the drift rate in ug/m3/day. The regional median is taken over all
    stations of the region at the same hour (robust to one bad sensor), so episodes and diurnal cycles cancel out.
    """
    cols = ["station_id", "drift_rate", "drift_tstat", "window_days", "drift_suspect"]
    if window.empty:
        res = pd.DataFrame({"station_id": stations["station_id"]})
        res["drift_rate"], res["drift_tstat"], res["window_days"], res["drift_suspect"] = np.nan, np.nan, 0, False
        res.insert(0, "date", date)
        return res
    w = window[["station_id", "region", "ts", "pm25"]].dropna()
    med = w.groupby(["region", "ts"])["pm25"].median().rename("reg_med")
    w = w.join(med, on=["region", "ts"])
    t0 = w["ts"].min().floor("D")
    w["t"] = (w["ts"] - t0) / pd.Timedelta(days=1)
    rows = []
    for sid, s in w.groupby("station_id"):
        n_days = s["ts"].dt.floor("D").nunique()
        if n_days < DRIFT_MIN_DAYS or len(s) < 48:
            rows.append((sid, np.nan, np.nan, n_days, False))
            continue
        X = np.column_stack([np.ones(len(s)), s["reg_med"].to_numpy(), s["t"].to_numpy()])
        y = s["pm25"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = max(1, len(y) - 3)
        sigma2 = float(resid @ resid) / dof
        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            se = float(np.sqrt(max(cov[2, 2], 1e-12)))
        except np.linalg.LinAlgError:
            se = np.inf
        g = float(beta[2])
        tstat = g / se if se > 0 else 0.0
        suspect = abs(g) >= DRIFT_MIN_SLOPE and abs(tstat) >= DRIFT_MIN_TSTAT
        rows.append((sid, round(g, 4), round(tstat, 2), n_days, bool(suspect)))
    res = pd.DataFrame(rows, columns=cols)
    res = stations[["station_id"]].merge(res, on="station_id", how="left")
    res["window_days"] = res["window_days"].fillna(0).astype(int)
    res["drift_suspect"] = res["drift_suspect"].astype("boolean").fillna(False).astype(bool)
    res.insert(0, "date", date)
    return res
