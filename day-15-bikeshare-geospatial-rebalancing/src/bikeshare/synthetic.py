"""Seeded SYNTHETIC trip generator over a real station layout (offline fallback).

Commute structure: residential stations send riders to job-dense stations in the morning and receive
them back in the evening; midday and weekend trips are short leisure hops. Weather damps demand.
Only used when the public Capital Bikeshare archive cannot be downloaded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .geo import haversine_matrix

CORE = (38.9007, -77.0365)  # downtown Washington (Farragut / McPherson Square)


def _gauss(h: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return np.exp(-0.5 * ((h - mu) / sd) ** 2)


def generate_trips(
    stations: pd.DataFrame, weather: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, seed: int = C.SEED
) -> pd.DataFrame:
    """Return SYNTHETIC trips with columns ride_id, start_time, end_time, start_id, end_id."""
    rng = np.random.default_rng(seed)
    lat = stations["lat"].to_numpy(float)
    lon = stations["lon"].to_numpy(float)
    ids = stations["station_id"].astype(str).to_numpy()
    n = len(ids)
    d_core = haversine_matrix(lat, lon, np.array([CORE[0]]), np.array([CORE[1]]))[:, 0] / 1000.0
    metro = stations.get("is_metro", pd.Series(False, index=stations.index)).to_numpy(bool)
    job = np.exp(-d_core / 2.5) + 0.3 * metro
    res = 1.0 - 0.7 * np.exp(-d_core / 2.5) + rng.uniform(0, 0.2, n)
    act = stations["activity_per_day"].to_numpy(float) if "activity_per_day" in stations else np.full(n, 20.0)
    dist = haversine_matrix(lat, lon, lat, lon) / 1000.0

    def dest_cdf(attract: np.ndarray, scale_km: float) -> np.ndarray:
        w = attract[None, :] * np.exp(-dist / scale_km)
        np.fill_diagonal(w, w.diagonal() * 0.15)
        return np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)

    cdfs = {"am": dest_cdf(job * act, 2.0), "pm": dest_cdf(res * act, 2.0), "off": dest_cdf(act, 1.5)}
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    weather = weather.reindex(hours).ffill().bfill()
    rows = []
    rate = act / 2.0  # mean daily departures per station
    for t, (temp, precip) in zip(hours, weather[["temp_c", "precip_mm"]].to_numpy(), strict=True):
        h = t.hour + 0.5
        weekend = t.dayofweek >= 5 or t.normalize() in C.HOLIDAYS
        if weekend:
            off, am, pm = 0.02 + 0.09 * _gauss(h, 14, 3.0), 0.0, 0.0
        else:
            off = 0.01 + 0.035 * _gauss(h, 12.5, 2.5) + 0.02 * _gauss(h, 20, 1.5)
            am, pm = 0.09 * _gauss(h, 8.2, 1.1), 0.09 * _gauss(h, 17.6, 1.4)
        wx = max(0.25, float(np.exp(-0.5 * precip))) * max(0.4, 1 - 0.03 * max(0.0, abs(temp - 22) - 6))
        for kind, weight in (("am", am * res * 1.6), ("pm", pm * job * 1.6), ("off", np.full(n, off))):
            if np.ndim(weight) == 0 and weight == 0:
                continue
            lam = rate * weight * wx
            k = rng.poisson(lam)
            if k.sum() == 0:
                continue
            orig = np.repeat(np.arange(n), k)
            u = rng.random(len(orig))
            dest = np.minimum((cdfs[kind][orig] < u[:, None]).sum(axis=1), n - 1)
            dur = 120 + dist[orig, dest] * 1.3 / 11.0 * 3600 * rng.lognormal(0, 0.3, len(orig))
            st = t + pd.to_timedelta(rng.uniform(0, 3600, len(orig)), unit="s")
            rows.append(
                pd.DataFrame(
                    {"start_time": st.floor("s"), "end_time": (st + pd.to_timedelta(dur, unit="s")).floor("s"), "o": orig, "d": dest}
                )
            )
    trips = pd.concat(rows, ignore_index=True).sort_values("start_time").reset_index(drop=True)
    trips["start_id"] = ids[trips.pop("o")]
    trips["end_id"] = ids[trips.pop("d")]
    trips.insert(0, "ride_id", [f"SYN{i:08d}" for i in range(len(trips))])
    return trips
