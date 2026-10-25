"""Hourly station panel: net flow target, day-ahead temporal lags, spatial lags over H3 neighbours."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config as C

STATIC_FEATURES = [
    "capacity",
    "metro_dist_m",
    "is_metro",
    "elevation_m",
    "elev_rel_ring2",
    "stations_in_cell",
    "stations_ring1",
    "stations_ring2",
    "poi_400m",
    "dist_core_m",
    "x_m",
    "y_m",
]
DYNAMIC_FEATURES = [
    "hour",
    "dow",
    "offday",
    "temp_c",
    "precip_mm",
    "precip_3h",
    "lag24",
    "lag48",
    "lag168",
    "roll7_same_hour",
    "ha_past_how",
    "dep_lag24",
    "arr_lag24",
    "dep_yday",
    "arr_yday",
    "nb_lag24",
    "nb_lag168",
    "nb_roll7",
    "nb_ha_past_how",
    "dep_roll7",
    "arr_roll7",
    "dep_ha_past_how",
    "arr_ha_past_how",
    "nb_dep_roll7",
    "nb_arr_roll7",
]
FEATURES = DYNAMIC_FEATURES + STATIC_FEATURES


def hourly_counts(
    trips: pd.DataFrame, station_ids: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Departures and arrivals per station-hour as (S, T) arrays."""
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    pos = pd.Series(np.arange(len(station_ids)), index=station_ids.astype(str).to_numpy())
    S, T = len(station_ids), len(hours)

    def count(times: pd.Series, ids: pd.Series) -> np.ndarray:
        t = ((times - start) // pd.Timedelta(hours=1)).to_numpy()
        s = ids.map(pos).to_numpy()
        ok = (t >= 0) & (t < T) & ~pd.isna(s)
        out = np.zeros((S, T), dtype=np.float32)
        np.add.at(out, (s[ok].astype(int), t[ok].astype(int)), 1.0)
        return out

    return count(trips["start_time"], trips["start_id"]), count(trips["end_time"], trips["end_id"]), hours


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float32)
    out[:, k:] = a[:, :-k]
    return out


def _past_same_how_mean(net: np.ndarray) -> np.ndarray:
    """Expanding mean of the same hour-of-week in previous weeks only (no leakage)."""
    S, T = net.shape
    s = np.zeros_like(net, dtype=np.float64)
    c = np.zeros(T)
    for t in range(168, T):
        s[:, t] = s[:, t - 168] + net[:, t - 168]
        c[t] = c[t - 168] + 1
    out = np.full_like(net, np.nan, dtype=np.float32)
    ok = c > 0
    out[:, ok] = (s[:, ok] / c[ok]).astype(np.float32)
    return out


def weights_matrix(neighbours: list[np.ndarray]) -> sp.csr_matrix:
    rows = np.concatenate([np.full(len(nb), i) for i, nb in enumerate(neighbours)])
    cols = np.concatenate(neighbours)
    vals = np.concatenate([np.full(len(nb), 1.0 / len(nb)) for nb in neighbours])
    n = len(neighbours)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def _nb(W: sp.csr_matrix, a: np.ndarray) -> np.ndarray:
    filled = np.nan_to_num(a, nan=0.0)
    out = (W @ filled).astype(np.float32)
    out[:, np.isnan(a).all(axis=0)] = np.nan
    return out


def build_panel(
    dep: np.ndarray, arr: np.ndarray, hours: pd.DatetimeIndex, stations: pd.DataFrame, weather: pd.DataFrame, neighbours: list[np.ndarray]
) -> pd.DataFrame:
    """Long station-hour table with the target ``net`` (arrivals - departures) and day-ahead features.

    Every temporal feature is at least 24 h old, so a forecast for day D is computable at 00:00 on D.
    Weather is the observed value (treated as a perfect day-ahead weather forecast).
    """
    S, T = dep.shape
    net = arr - dep
    W = weights_matrix(neighbours)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        roll7 = np.nanmean(np.stack([_shift(net, 24 * k) for k in range(1, 8)]), axis=0)
    ha_past = _past_same_how_mean(net)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dep_roll7 = np.nanmean(np.stack([_shift(dep, 24 * k) for k in range(1, 8)]), axis=0)
        arr_roll7 = np.nanmean(np.stack([_shift(arr, 24 * k) for k in range(1, 8)]), axis=0)
    day_idx = (hours - hours[0]).days.to_numpy()
    n_days = day_idx.max() + 1
    dep_day = np.stack([dep[:, day_idx == d].sum(axis=1) for d in range(n_days)], axis=1)
    arr_day = np.stack([arr[:, day_idx == d].sum(axis=1) for d in range(n_days)], axis=1)
    dep_yday = np.full((S, T), np.nan, dtype=np.float32)
    arr_yday = np.full((S, T), np.nan, dtype=np.float32)
    ok = day_idx >= 1
    dep_yday[:, ok] = dep_day[:, day_idx[ok] - 1]
    arr_yday[:, ok] = arr_day[:, day_idx[ok] - 1]

    w = weather.reindex(hours)
    precip3 = w["precip_mm"].rolling(3, min_periods=1).sum().to_numpy()
    lag24 = _shift(net, 24)
    lag168 = _shift(net, 168)
    offday = (hours.dayofweek >= 5) | hours.normalize().isin(list(C.HOLIDAYS))

    cols: dict[str, np.ndarray] = {
        "station": np.repeat(np.arange(S), T),
        "t": np.tile(np.arange(T), S),
        "net": net.ravel(),
        "dep": dep.ravel(),
        "arr": arr.ravel(),
        "hour": np.tile(hours.hour.to_numpy(), S),
        "dow": np.tile(hours.dayofweek.to_numpy(), S),
        "offday": np.tile(offday.astype(np.int8), S),
        "temp_c": np.tile(w["temp_c"].to_numpy(np.float32), S),
        "precip_mm": np.tile(w["precip_mm"].to_numpy(np.float32), S),
        "precip_3h": np.tile(precip3.astype(np.float32), S),
        "lag24": lag24.ravel(),
        "lag48": _shift(net, 48).ravel(),
        "lag168": lag168.ravel(),
        "roll7_same_hour": roll7.ravel(),
        "ha_past_how": ha_past.ravel(),
        "dep_lag24": _shift(dep, 24).ravel(),
        "arr_lag24": _shift(arr, 24).ravel(),
        "dep_yday": dep_yday.ravel(),
        "arr_yday": arr_yday.ravel(),
        "nb_lag24": _nb(W, lag24).ravel(),
        "nb_lag168": _nb(W, lag168).ravel(),
        "nb_roll7": _nb(W, roll7).ravel(),
        "nb_ha_past_how": _nb(W, ha_past).ravel(),
        "dep_roll7": dep_roll7.ravel(),
        "arr_roll7": arr_roll7.ravel(),
        "dep_ha_past_how": _past_same_how_mean(dep).ravel(),
        "arr_ha_past_how": _past_same_how_mean(arr).ravel(),
        "nb_dep_roll7": _nb(W, dep_roll7).ravel(),
        "nb_arr_roll7": _nb(W, arr_roll7).ravel(),
    }
    panel = pd.DataFrame(cols)
    st = stations.reset_index(drop=True)
    for f in STATIC_FEATURES:
        panel[f] = st[f].to_numpy(np.float32)[panel["station"].to_numpy()]
    panel["time"] = hours[panel["t"].to_numpy()]
    return panel
