"""Feature table for the global LightGBM model.

Forecasts are issued at 23:00 for the 24 hours of the next day, so the target at time t has horizon
h = hour(t) + 1 and the freshest observed load is at t - h. Every load feature therefore uses only
values at or before t - 24, which is never later than the forecast origin. Temperature enters only as
the day-ahead *forecast* (temp_fcst_c), never the realised value at the target hour.

The target is load divided by a trailing 7-day mean ("level"), which makes one model work across
regions of different size and lets it follow level shifts (the CENTRAL data-centre regime) instead of
being capped by the range of loads seen in training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import REGIONS

LOAD_LAGS = (24, 48, 72, 168, 336)

FEATURES: list[str] = [
    "region_code",
    "hour",
    "dow",
    "is_holiday",
    "is_offday",
    "is_xmas_week",
    "doy_sin",
    "doy_cos",
    *[f"lag{k}_rel" for k in LOAD_LAGS],
    "day_mean_rel",
    "level_log",
    "temp_fcst_c",
    "temp_fcst_smooth",
    "hdd_fcst",
    "cdd_fcst",
    "temp_lag24",
    "temp_fcst_day_max",
    "temp_fcst_day_min",
]
CATEGORICAL = ["region_code"]
REGION_CODE = {r.name: i for i, r in enumerate(REGIONS)}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model features to a long frame of region, ds, load_mw, temp_c, temp_fcst_c, is_holiday.

    Returns one row per input row (rows without enough history have NaN features and are dropped by callers).
    """
    out = []
    for region, g in df.sort_values(["region", "ds"]).groupby("region", sort=False):
        g = g.copy()
        y = g["load_mw"]
        # Trailing 7-day mean ending at t-24 (known at the origin for every horizon).
        level = y.shift(24).rolling(168, min_periods=168).mean()
        g["level"] = level
        g["level_log"] = np.log(level)
        for k in LOAD_LAGS:
            g[f"lag{k}_rel"] = y.shift(k) / level
        g["day_mean_rel"] = y.shift(24).rolling(24, min_periods=24).mean() / level
        g["y_rel"] = y / level

        tf = g["temp_fcst_c"]
        g["temp_fcst_smooth"] = tf.ewm(halflife=5, adjust=False).mean()
        g["hdd_fcst"] = np.maximum(0.0, 15.0 - g["temp_fcst_smooth"])
        g["cdd_fcst"] = np.maximum(0.0, g["temp_fcst_smooth"] - 20.0)
        g["temp_lag24"] = g["temp_c"].shift(24)
        day = g["ds"].dt.normalize()
        g["temp_fcst_day_max"] = tf.groupby(day).transform("max")
        g["temp_fcst_day_min"] = tf.groupby(day).transform("min")
        g["region_code"] = REGION_CODE[region]
        out.append(g)
    f = pd.concat(out, ignore_index=True)
    ds = f["ds"]
    f["hour"] = ds.dt.hour
    f["horizon"] = f["hour"] + 1
    f["dow"] = ds.dt.dayofweek
    f["is_offday"] = ((f["dow"] >= 5) | (f["is_holiday"] == 1)).astype(np.int8)
    f["is_xmas_week"] = (((ds.dt.month == 12) & (ds.dt.day >= 24)) | ((ds.dt.month == 1) & (ds.dt.day <= 1))).astype(
        np.int8
    )
    doy = ds.dt.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return f
