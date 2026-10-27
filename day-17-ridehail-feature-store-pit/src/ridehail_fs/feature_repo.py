"""Feast repository: entities, feature views, an on-demand view and the model's feature service.

The store is fully local: DuckDB offline store over parquet, SQLite online
store, file registry. ``feature_repo/feature_store.redis.yaml`` shows the
server-backed variant (Redis online store); nothing here needs it.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from feast import (
    Entity,
    FeatureService,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RepoConfig,
    RequestSource,
    ValueType,
)
from feast.infra.offline_stores.duckdb import DuckDBOfflineStoreConfig
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, Int64

PROJECT = "ridehail_eta"
ZONE_TTL = timedelta(minutes=30)
DRIVER_TTL = timedelta(hours=3)

driver = Entity(name="driver", join_keys=["driver_id"], value_type=ValueType.INT64, description="A driver on the platform")
zone = Entity(name="zone", join_keys=["zone_id"], value_type=ValueType.INT64, description="Pickup zone (2 km grid cell)")
dropoff_zone = Entity(name="dropoff_zone", join_keys=["dropoff_zone_id"], value_type=ValueType.INT64, description="Drop-off zone")


def eta_transform(inputs: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Request-time features. Pure python so it is cheap online and identical offline.

    Feast stores this function's source in the registry and re-executes it, so imports live inside.
    """
    import math

    out: dict[str, list[Any]] = {k: [] for k in ("eta_heuristic_min", "driver_adjusted_eta_min", "hour_sin", "hour_cos", "is_weekend")}
    for dist, v, slow, hr, dow in zip(
        inputs["distance_km"], inputs["avg_speed_15m"], inputs["slowness_ratio_7d"], inputs["request_hour"], inputs["day_of_week"], strict=True
    ):
        speed = v if (v is not None and v == v and v > 1.0) else 25.0  # NaN/None -> city-wide default
        eta = float(dist) / speed * 60.0
        s = slow if (slow is not None and slow == slow) else 1.0
        out["eta_heuristic_min"].append(eta)
        out["driver_adjusted_eta_min"].append(eta * min(max(s / 1.6, 0.6), 1.6))
        out["hour_sin"].append(math.sin(2 * math.pi * float(hr) / 24.0))
        out["hour_cos"].append(math.cos(2 * math.pi * float(hr) / 24.0))
        out["is_weekend"].append(int(int(dow) >= 5))
    return out


def _fresh(fn: Any) -> Any:
    """Copy of ``fn``: Feast's decorator rewrites ``__module__`` in place, which breaks a second
    ``dill.source.getsource`` call (tests build several stores in one process)."""
    import types

    new = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
    new.__module__ = __name__
    new.__doc__ = fn.__doc__
    return new


def build_definitions(offline_dir: Path) -> dict[str, Any]:
    """Create all Feast objects pointing at the parquet tables in ``offline_dir``."""
    offline_dir = Path(offline_dir).resolve()

    def src(name: str) -> FileSource:
        return FileSource(
            name=f"{name}_source",
            path=str(offline_dir / f"{name}.parquet"),
            timestamp_field="event_timestamp",
            created_timestamp_column="created",
        )

    zone_traffic_src = src("zone_traffic")
    zone_traffic_push = PushSource(name="zone_traffic_push", batch_source=zone_traffic_src)
    zone_traffic = FeatureView(
        name="zone_traffic",
        entities=[zone],
        ttl=ZONE_TTL,
        schema=[
            Field(name="zone_id", dtype=Int64),
            Field(name="avg_speed_15m", dtype=Float64, description="Mean GPS-probe speed, km/h, trailing 15 min"),
            Field(name="avg_speed_60m", dtype=Float64, description="Mean GPS-probe speed, km/h, trailing 60 min"),
            Field(name="probe_count_15m", dtype=Float64),
            Field(name="speed_trend", dtype=Float64, description="15-min / 60-min speed"),
        ],
        source=zone_traffic_push,
        online=True,
        tags={"team": "marketplace", "cadence": "5min"},
    )
    zone_demand = FeatureView(
        name="zone_demand",
        entities=[zone],
        ttl=ZONE_TTL,
        schema=[
            Field(name="zone_id", dtype=Int64),
            Field(name="requests_15m", dtype=Float64),
            Field(name="requests_60m", dtype=Float64),
            Field(name="demand_index", dtype=Float64, description="(4*req_15m+1)/(req_60m+1)"),
        ],
        source=src("zone_demand"),
        online=True,
        tags={"team": "marketplace", "cadence": "5min"},
    )
    driver_stats = FeatureView(
        name="driver_stats",
        entities=[driver],
        ttl=DRIVER_TTL,
        schema=[
            Field(name="driver_id", dtype=Int64),
            Field(name="acceptance_rate_7d", dtype=Float64),
            Field(name="trips_24h", dtype=Float64),
            Field(name="slowness_ratio_7d", dtype=Float64, description="Mean actual/free-flow duration of trips completed in 7 d"),
        ],
        source=src("driver_stats"),
        online=True,
        tags={"team": "supply", "cadence": "1h"},
    )
    request = RequestSource(
        name="eta_request",
        schema=[
            Field(name="distance_km", dtype=Float64),
            Field(name="request_hour", dtype=Float64),
            Field(name="day_of_week", dtype=Int64),
        ],
    )
    eta_features = on_demand_feature_view(
        name="eta_request_features",
        sources=[zone_traffic, driver_stats, request],
        schema=[
            Field(name="eta_heuristic_min", dtype=Float64),
            Field(name="driver_adjusted_eta_min", dtype=Float64),
            Field(name="hour_sin", dtype=Float64),
            Field(name="hour_cos", dtype=Float64),
            Field(name="is_weekend", dtype=Int64),
        ],
        mode="python",
        description="Request-time transforms; the same code runs for training (offline) and serving (online)",
    )(_fresh(eta_transform))
    # Same table, keyed by the drop-off zone. A separate view (rather than a projection alias) keeps feature names unique.
    dropoff_src = FileSource(
        name="dropoff_traffic_source",
        path=str(offline_dir / "zone_traffic.parquet"),
        timestamp_field="event_timestamp",
        created_timestamp_column="created",
        field_mapping={"zone_id": "dropoff_zone_id", "avg_speed_15m": "dropoff_avg_speed_15m"},
    )
    dropoff_traffic = FeatureView(
        name="dropoff_traffic",
        entities=[dropoff_zone],
        ttl=ZONE_TTL,
        schema=[Field(name="dropoff_zone_id", dtype=Int64), Field(name="dropoff_avg_speed_15m", dtype=Float64)],
        source=dropoff_src,
        online=True,
        tags={"team": "marketplace", "cadence": "5min"},
    )
    service = FeatureService(
        name="eta_model_v1",
        features=[zone_traffic, zone_demand, driver_stats, dropoff_traffic, eta_features],
        description="Every feature the ETA LightGBM model consumes",
    )
    return {
        "entities": [driver, zone, dropoff_zone],
        "sources": [zone_traffic_src, zone_traffic_push, dropoff_src],
        "feature_views": [zone_traffic, zone_demand, driver_stats, dropoff_traffic],
        "odfvs": [eta_features],
        "service": service,
        "request_source": request,
    }


def make_store(store_dir: Path, offline_dir: Path) -> tuple[FeatureStore, dict[str, Any]]:
    """Build a local FeatureStore (DuckDB offline, SQLite online) and apply the definitions."""
    store_dir = Path(store_dir).resolve()
    store_dir.mkdir(parents=True, exist_ok=True)
    config = RepoConfig(
        project=PROJECT,
        provider="local",
        registry=str(store_dir / "registry.db"),
        offline_store=DuckDBOfflineStoreConfig(),
        online_store=SqliteOnlineStoreConfig(path=str(store_dir / "online.db")),
        entity_key_serialization_version=3,
    )
    store = FeatureStore(config=config, repo_path=str(store_dir))
    defs = build_definitions(offline_dir)
    store.apply([*defs["entities"], *defs["sources"], *defs["feature_views"], *defs["odfvs"], defs["service"]])
    return store, defs
