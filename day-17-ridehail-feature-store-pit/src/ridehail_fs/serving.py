"""Online side: materialisation replay, training/serving skew test, latency benchmark, push."""

from __future__ import annotations

import time
from datetime import timedelta

import numpy as np
import pandas as pd
from feast import FeatureStore

from .feature_repo import DRIVER_TTL
from .joins import ALL_FEATURES, compare_frames, entity_frame


def online_rows(ent: pd.DataFrame) -> list[dict]:
    """Serving requests: entity keys + request data (no timestamps: online means 'now')."""
    return [
        {
            "driver_id": int(r.driver_id),
            "zone_id": int(r.zone_id),
            "dropoff_zone_id": int(r.dropoff_zone_id),
            "distance_km": float(r.distance_km),
            "request_hour": float(r.request_hour),
            "day_of_week": int(r.day_of_week),
        }
        for r in ent.itertuples(index=False)
    ]


def fetch_online(store: FeatureStore, ent: pd.DataFrame) -> pd.DataFrame:
    service = store.get_feature_service("eta_model_v1")
    out = store.get_online_features(features=service, entity_rows=online_rows(ent)).to_df()
    out.insert(0, "trip_id", ent["trip_id"].to_numpy())
    return out


def checkpoint_times(start: pd.Timestamp, end: pd.Timestamp, n: int, seed: int, tick_min: int = 5) -> list[pd.Timestamp]:
    """n random 5-minute boundaries (+1 s) in [start, end), excluding windows that straddle an hour boundary."""
    rng = np.random.default_rng(seed)
    grid = pd.date_range(start.ceil(f"{tick_min}min"), end - pd.Timedelta(minutes=tick_min), freq=f"{tick_min}min")
    grid = grid[grid.minute != 60 - tick_min]  # the next driver row (hourly) must not land inside the window
    pick = np.sort(rng.choice(len(grid), size=min(n, len(grid)), replace=False))
    return [grid[i] + pd.Timedelta(seconds=1) for i in pick]


def replay_online(store: FeatureStore, trips: pd.DataFrame, checkpoints: list[pd.Timestamp], window_s: int = 239) -> tuple[pd.DataFrame, dict]:
    """Simulate production: at each checkpoint T, materialise everything known by T into the
    online store, then serve the requests that arrive in (T, T + 4 min) from it.

    Feature rows land every 5 min (zones) or every hour (drivers), so nothing newer than T
    exists for those requests: the online answer must equal the offline PIT answer.
    """
    frames, mat_s = [], []
    for T in checkpoints:
        reqs = trips[(trips.request_ts > T) & (trips.request_ts <= T + pd.Timedelta(seconds=window_s))]
        if reqs.empty:
            continue
        t0 = time.perf_counter()
        store.materialize(start_date=(T - DRIVER_TTL - timedelta(minutes=5)).to_pydatetime(), end_date=T.to_pydatetime())
        mat_s.append(time.perf_counter() - t0)
        frames.append(fetch_online(store, entity_frame(reqs)).assign(checkpoint=T))
    online = pd.concat(frames, ignore_index=True)
    return online, {"checkpoints": len(frames), "materialize_s_mean": float(np.mean(mat_s)), "materialize_s_total": float(np.sum(mat_s))}


def skew_report(online: pd.DataFrame, offline_pit: pd.DataFrame) -> dict:
    """Training/serving skew: every feature the online store served vs what training used."""
    off = offline_pit[offline_pit.trip_id.isin(online.trip_id)]
    return compare_frames(off, online, ALL_FEATURES)


def _epoch_s(x: pd.Series) -> np.ndarray:
    return (pd.to_datetime(x, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def handrolled_speed(probes: pd.DataFrame, ent: pd.DataFrame, minutes: int = 15) -> np.ndarray:
    """A re-implementation a serving team might write: mean probe speed over the 15 minutes
    before the request *second*, straight from the raw stream (not the 5-min aligned windows)."""
    p = probes.sort_values(["zone_id", "probe_ts"])
    by_zone = {z: (_epoch_s(g.probe_ts), g.speed_kmh.to_numpy()) for z, g in p.groupby("zone_id")}
    t_req = _epoch_s(ent.event_timestamp)
    out = np.full(len(ent), np.nan)
    for i, (z, t) in enumerate(zip(ent.zone_id.to_numpy(), t_req, strict=True)):
        if z not in by_zone:
            continue
        ts, sp = by_zone[z]
        lo, hi = np.searchsorted(ts, t - minutes * 60, "left"), np.searchsorted(ts, t, "left")
        if hi > lo:
            out[i] = sp[lo:hi].mean()
    return out


def latency_benchmark(store: FeatureStore, ent: pd.DataFrame, n: int = 2000, warmup: int = 100, batch: int = 50, seed: int = 0) -> dict:
    """Per-request latency of get_online_features (feature service incl. on-demand transform)."""
    rng = np.random.default_rng(seed)
    rows = online_rows(ent)
    service = store.get_feature_service("eta_model_v1")
    idx = rng.integers(0, len(rows), n + warmup)
    lat = []
    for k, i in enumerate(idx):
        t0 = time.perf_counter()
        store.get_online_features(features=service, entity_rows=[rows[i]]).to_dict()
        if k >= warmup:
            lat.append((time.perf_counter() - t0) * 1000.0)
    blat = []
    for _ in range(200):
        sel = [rows[j] for j in rng.integers(0, len(rows), batch)]
        t0 = time.perf_counter()
        store.get_online_features(features=service, entity_rows=sel).to_dict()
        blat.append((time.perf_counter() - t0) * 1000.0)
    lat, blat = np.array(lat), np.array(blat)
    return {
        "single_row": {
            "n": int(len(lat)),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)),
            "max_ms": float(lat.max()),
        },
        f"batch_{batch}": {"n": int(len(blat)), "p50_ms": float(np.percentile(blat, 50)), "p99_ms": float(np.percentile(blat, 99))},
        "latencies_ms": lat,
    }


def push_demo(store: FeatureStore, zone_id: int, ts: pd.Timestamp, speed: float) -> dict:
    """Streaming stretch goal: push one fresh zone_traffic row and read it back online."""
    before = store.get_online_features(features=["zone_traffic:avg_speed_15m"], entity_rows=[{"zone_id": zone_id}]).to_dict()["avg_speed_15m"][0]
    row = pd.DataFrame(
        {
            "zone_id": [zone_id],
            "event_timestamp": [ts],
            "created": [ts],
            "avg_speed_15m": [speed],
            "avg_speed_60m": [speed],
            "probe_count_15m": [1.0],
            "speed_trend": [1.0],
        }
    )
    t0 = time.perf_counter()
    store.push("zone_traffic_push", row)
    push_ms = (time.perf_counter() - t0) * 1000.0
    after = store.get_online_features(features=["zone_traffic:avg_speed_15m"], entity_rows=[{"zone_id": zone_id}]).to_dict()["avg_speed_15m"][0]
    return {"zone_id": zone_id, "before": before, "pushed": speed, "after": after, "push_ms": push_ms, "visible_online": bool(abs(after - speed) < 1e-9)}
