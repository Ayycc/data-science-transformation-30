"""US EPA Air Quality Index (2024 PM2.5 breakpoints) and daily station AQI aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd

# (C_lo, C_hi, I_lo, I_hi). Concentrations are truncated before lookup as EPA prescribes.
BREAKPOINTS: dict[str, list[tuple[float, float, int, int]]] = {
    # 24-hour mean, ug/m3, 1 decimal
    "pm25": [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
    ],
    # daily max 1-hour, ppb, integer
    "no2": [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500),
    ],
    # daily max 8-hour mean, ppm, 3 decimals (8-hour O3 is not defined above 0.200 ppm: capped at 300)
    "o3": [
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ],
}
TRUNC_DECIMALS = {"pm25": 1, "no2": 0, "o3": 3}

CATEGORIES = [
    (50, "Good"),
    (100, "Moderate"),
    (150, "Unhealthy for Sensitive Groups"),
    (200, "Unhealthy"),
    (300, "Very Unhealthy"),
    (10_000, "Hazardous"),
]


def _truncate(x: np.ndarray, decimals: int) -> np.ndarray:
    f = 10.0**decimals
    # small epsilon guards against binary representation (e.g. 35.4 stored as 35.39999)
    return np.floor(x * f + 1e-9) / f


def sub_index(pollutant: str, conc: np.ndarray | pd.Series | float) -> np.ndarray:
    """AQI sub-index for concentrations in the averaging unit of that pollutant (NaN stays NaN)."""
    c = _truncate(np.asarray(conc, dtype=float), TRUNC_DECIMALS[pollutant])
    bp = BREAKPOINTS[pollutant]
    lo = np.array([b[0] for b in bp])
    out = np.full(c.shape, np.nan)
    ok = ~np.isnan(c)
    idx = np.clip(np.searchsorted(lo, c[ok], side="right") - 1, 0, len(bp) - 1)
    clo = lo[idx]
    chi = np.array([b[1] for b in bp])[idx]
    ilo = np.array([b[2] for b in bp])[idx]
    ihi = np.array([b[3] for b in bp])[idx]
    val = (ihi - ilo) / (chi - clo) * (np.minimum(c[ok], chi) - clo) + ilo
    # Above the top breakpoint: 500 for PM2.5/NO2, 300 cap for 8-hour O3.
    top = c[ok] > bp[-1][1]
    val = np.where(top, float(bp[-1][3]), val)
    out[ok] = np.floor(val + 0.5)
    return out


def category(aqi: np.ndarray | pd.Series) -> np.ndarray:
    a = np.asarray(aqi, dtype=float)
    labels = np.array([c[1] for c in CATEGORIES] + [None], dtype=object)
    edges = np.array([c[0] for c in CATEGORIES], dtype=float)
    idx = np.searchsorted(edges, a, side="left")
    res = labels[np.clip(idx, 0, len(CATEGORIES) - 1)]
    res[np.isnan(a)] = None
    return res


def daily_station_aqi(hourly: pd.DataFrame, min_hours: int = 18) -> pd.DataFrame:
    """Daily AQI per station from hourly values (columns station_id, ts, pm25, no2, o3; NaN = invalid).

    PM2.5: 24-h mean of valid hours (>= ``min_hours``). NO2: daily max 1-h (>= ``min_hours`` valid).
    O3: max of 8-h running means fully inside the day with >= 6 valid hours, ppb -> ppm.
    """
    df = hourly.copy()
    df["date"] = df["ts"].dt.floor("D")
    g = df.groupby(["station_id", "date"], sort=True)
    agg = g.agg(
        pm25_hours=("pm25", "count"),
        pm25_mean=("pm25", "mean"),
        no2_hours=("no2", "count"),
        no2_max=("no2", "max"),
        o3_hours=("o3", "count"),
    ).reset_index()

    # 8-hour O3 on a complete hourly grid.
    full = df.set_index(["station_id", "date", df["ts"].dt.hour.rename("hour")])["o3"]
    full = full[~full.index.duplicated(keep="last")].unstack("hour")
    full = full.reindex(columns=range(24))
    arr = full.to_numpy(dtype=float)
    win = np.lib.stride_tricks.sliding_window_view(arr, 8, axis=1)  # (n, 17, 8)
    cnt = np.sum(~np.isnan(win), axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(cnt >= 6, np.nansum(win, axis=2) / np.maximum(cnt, 1), np.nan)
    with np.errstate(all="ignore"):
        o3_8h = np.where(
            np.all(np.isnan(means), axis=1), np.nan, np.nanmax(np.where(np.isnan(means), -np.inf, means), axis=1)
        )
    o3_8h = np.where(np.isinf(o3_8h), np.nan, o3_8h)
    o3_frame = pd.DataFrame({"o3_8h_max": o3_8h}, index=full.index).reset_index()
    agg = agg.merge(o3_frame, on=["station_id", "date"], how="left")

    agg.loc[agg["pm25_hours"] < min_hours, "pm25_mean"] = np.nan
    agg.loc[agg["no2_hours"] < min_hours, "no2_max"] = np.nan
    agg["aqi_pm25"] = sub_index("pm25", agg["pm25_mean"])
    agg["aqi_no2"] = sub_index("no2", agg["no2_max"])
    agg["aqi_o3"] = sub_index("o3", agg["o3_8h_max"] / 1000.0)
    subs = agg[["aqi_pm25", "aqi_no2", "aqi_o3"]].to_numpy()
    all_nan = np.all(np.isnan(subs), axis=1)
    agg["aqi"] = np.where(all_nan, np.nan, np.nanmax(np.where(np.isnan(subs), -1, subs), axis=1))
    dom = np.array(["pm25", "no2", "o3"], dtype=object)[np.argmax(np.where(np.isnan(subs), -1, subs), axis=1)]
    agg["dominant_pollutant"] = np.where(all_nan, None, dom)
    agg["category"] = category(agg["aqi"])
    return agg
