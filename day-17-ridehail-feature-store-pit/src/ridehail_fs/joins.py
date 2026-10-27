"""Three ways to attach features to trips.

1. ``naive_join``      - what the failing pipeline did: join warehouse roll-ups on (zone, clock hour)
                         and a driver profile computed over the whole export. Leaks the future.
2. ``feast_pit_join``  - Feast ``get_historical_features``: for each request, the latest feature row
                         stamped at or before the request time and within the view's TTL.
3. ``reference_asof``  - an independent DuckDB ASOF JOIN with the same rule, used to check Feast.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from feast import FeatureStore

from .feature_repo import DRIVER_TTL, ZONE_TTL, eta_transform
from .features import DRIVER_FEATURES, ZONE_DEMAND_FEATURES, ZONE_TRAFFIC_FEATURES

ODFV_FEATURES = ["eta_heuristic_min", "driver_adjusted_eta_min", "hour_sin", "hour_cos", "is_weekend"]
STORE_FEATURES = [*ZONE_TRAFFIC_FEATURES, *ZONE_DEMAND_FEATURES, *DRIVER_FEATURES, "dropoff_avg_speed_15m"]
ALL_FEATURES = [*STORE_FEATURES, *ODFV_FEATURES]


def entity_frame(trips: pd.DataFrame) -> pd.DataFrame:
    """Entity rows + request data, exactly as the serving request would carry them."""
    ts = trips["request_ts"]
    return pd.DataFrame(
        {
            "trip_id": trips["trip_id"].to_numpy(),
            "driver_id": trips["driver_id"].to_numpy(),
            "zone_id": trips["zone_id"].to_numpy(),
            "dropoff_zone_id": trips["dropoff_zone_id"].to_numpy(),
            "distance_km": trips["distance_km"].to_numpy(dtype=float),
            "event_timestamp": ts.to_numpy(),
            "request_hour": (ts.dt.hour + ts.dt.minute / 60.0).to_numpy(dtype=float),
            "day_of_week": ts.dt.dayofweek.to_numpy(dtype=np.int64),
        }
    )


def apply_odfv(df: pd.DataFrame) -> pd.DataFrame:
    """Run the same on-demand transform Feast runs, on an already-joined frame."""
    cols = ["distance_km", "avg_speed_15m", "slowness_ratio_7d", "request_hour", "day_of_week"]
    inputs = {c: df[c].astype(object).where(df[c].notna(), None).tolist() for c in cols}
    out = pd.DataFrame(eta_transform(inputs), index=df.index)
    return pd.concat([df.drop(columns=[c for c in ODFV_FEATURES if c in df.columns]), out], axis=1)


def naive_join(trips: pd.DataFrame, snapshots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The leaky pipeline: hour-bucket equality joins against end-of-hour roll-ups + whole-export driver profile."""
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.register("ent", entity_frame(trips))
    for k, v in snapshots.items():
        con.register(k, v)
    zt = ", ".join(f"zt.{c}" for c in ZONE_TRAFFIC_FEATURES)
    zd = ", ".join(f"zd.{c}" for c in ZONE_DEMAND_FEATURES)
    dp = ", ".join(f"dp.{c}" for c in DRIVER_FEATURES)
    df = con.execute(
        f"""
        SELECT e.*, {zt}, {zd}, {dp}, dz.avg_speed_15m AS dropoff_avg_speed_15m
        FROM ent e
        LEFT JOIN zone_traffic_hourly zt ON zt.zone_id = e.zone_id AND zt.hour = date_trunc('hour', e.event_timestamp)
        LEFT JOIN zone_demand_hourly zd ON zd.zone_id = e.zone_id AND zd.hour = date_trunc('hour', e.event_timestamp)
        LEFT JOIN zone_traffic_hourly dz ON dz.zone_id = e.dropoff_zone_id AND dz.hour = date_trunc('hour', e.event_timestamp)
        LEFT JOIN driver_profile dp ON dp.driver_id = e.driver_id
        ORDER BY e.trip_id
        """
    ).df()
    con.close()
    return apply_odfv(df)


def feast_pit_join(store: FeatureStore, trips: pd.DataFrame, chunk: int = 50_000) -> pd.DataFrame:
    """Point-in-time join through Feast's DuckDB offline store, using the model's feature service."""
    service = store.get_feature_service("eta_model_v1")
    ent = entity_frame(trips)
    parts = []
    for i in range(0, len(ent), chunk):
        parts.append(store.get_historical_features(entity_df=ent.iloc[i : i + chunk], features=service).to_df())
    out = pd.concat(parts, ignore_index=True).sort_values("trip_id").reset_index(drop=True)
    out["event_timestamp"] = pd.to_datetime(out["event_timestamp"], utc=True)
    return out


def reference_asof(trips: pd.DataFrame, feats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Independent PIT implementation (DuckDB ASOF JOIN + TTL). Also returns each feature row's age."""
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.register("ent", entity_frame(trips))
    for k, v in feats.items():
        con.register(k, v)
    zttl, dttl = int(ZONE_TTL.total_seconds()), int(DRIVER_TTL.total_seconds())

    def cols(alias: str, names: list[str], ttl: int, rename: dict[str, str] | None = None) -> str:
        rename = rename or {}
        return ", ".join(
            f"CASE WHEN e.event_timestamp - {alias}.event_timestamp <= INTERVAL {ttl} SECOND THEN {alias}.{c} END AS {rename.get(c, c)}" for c in names
        )

    df = con.execute(
        f"""
        SELECT e.*,
          {cols("zt", ZONE_TRAFFIC_FEATURES, zttl)},
          {cols("zd", ZONE_DEMAND_FEATURES, zttl)},
          {cols("ds", DRIVER_FEATURES, dttl)},
          {cols("dz", ["avg_speed_15m"], zttl, {"avg_speed_15m": "dropoff_avg_speed_15m"})},
          epoch(e.event_timestamp - zt.event_timestamp) / 60.0 AS zone_feature_age_min,
          epoch(e.event_timestamp - ds.event_timestamp) / 60.0 AS driver_feature_age_min
        FROM ent e
        ASOF LEFT JOIN zone_traffic zt ON zt.zone_id = e.zone_id AND zt.event_timestamp <= e.event_timestamp
        ASOF LEFT JOIN zone_demand zd ON zd.zone_id = e.zone_id AND zd.event_timestamp <= e.event_timestamp
        ASOF LEFT JOIN driver_stats ds ON ds.driver_id = e.driver_id AND ds.event_timestamp <= e.event_timestamp
        ASOF LEFT JOIN zone_traffic dz ON dz.zone_id = e.dropoff_zone_id AND dz.event_timestamp <= e.event_timestamp
        ORDER BY e.trip_id
        """
    ).df()
    con.close()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    return apply_odfv(df)


def compare_frames(a: pd.DataFrame, b: pd.DataFrame, cols: list[str], key: str = "trip_id", rtol: float = 1e-9) -> dict:
    """Cell-level equality of two feature frames (NaN == NaN counts as equal)."""
    a = a.set_index(key).sort_index()
    b = b.set_index(key).loc[a.index]
    mism, per_col = 0, {}
    for c in cols:
        x = pd.to_numeric(a[c], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(b[c], errors="coerce").to_numpy(dtype=float)
        bad = ~(np.isclose(x, y, rtol=rtol, atol=1e-9) | (np.isnan(x) & np.isnan(y)))
        per_col[c] = int(bad.sum())
        mism += int(bad.sum())
    return {"rows": int(len(a)), "cells": int(len(a) * len(cols)), "mismatched_cells": mism, "per_column": per_col}
