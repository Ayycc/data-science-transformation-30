"""Offline feature tables, computed in DuckDB from the raw event tables.

Each feature row is stamped with the end of the window it summarises
(``event_timestamp``): a row stamped 08:15 only uses events strictly before
08:15. That stamp is what makes a point-in-time join possible.

- ``zone_traffic``  : every 5 min per zone - rolling GPS-probe speed (15/60 min), probe count, trend
- ``zone_demand``   : every 5 min per zone - rolling ride requests (15/60 min) and a demand index
- ``driver_stats``  : every hour per driver - 7-day acceptance rate, trips in 24 h, 7-day slowness ratio
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ZONE_TRAFFIC_FEATURES = ["avg_speed_15m", "avg_speed_60m", "probe_count_15m", "speed_trend"]
ZONE_DEMAND_FEATURES = ["requests_15m", "requests_60m", "demand_index"]
DRIVER_FEATURES = ["acceptance_rate_7d", "trips_24h", "slowness_ratio_7d"]


def _grid_bounds(trips: pd.DataFrame, tick_min: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    lo = trips.request_ts.min().floor("D")
    hi = trips.dropoff_ts.max().ceil(f"{tick_min}min") + pd.Timedelta(minutes=tick_min)
    return lo, hi


def build_feature_tables(tables: dict[str, pd.DataFrame], tick_min: int = 5) -> dict[str, pd.DataFrame]:
    """Compute the three event-timestamped feature tables with DuckDB window functions."""
    trips, offers, probes = tables["trips"], tables["offers"], tables["probes"]
    lo, hi = _grid_bounds(trips, tick_min)
    zones = sorted(set(trips.zone_id) | set(trips.dropoff_zone_id) | set(probes.zone_id))
    drivers = sorted(set(trips.driver_id) | set(offers.driver_id))
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.register("trips", trips)
    con.register("offers", offers)
    con.register("probes", probes)
    con.register("zones", pd.DataFrame({"zone_id": zones}))
    con.register("drivers", pd.DataFrame({"driver_id": drivers}))
    step = f"INTERVAL {tick_min} MINUTE"
    lo_s, hi_s = lo.strftime("%Y-%m-%d %H:%M:%S+00"), hi.strftime("%Y-%m-%d %H:%M:%S+00")
    con.execute(
        f"""
        CREATE TABLE ticks AS
        SELECT unnest(generate_series(TIMESTAMPTZ '{lo_s}', TIMESTAMPTZ '{hi_s}', {step})) AS tick_start
        """
    )
    zone_traffic = con.execute(
        f"""
        WITH agg AS (
            SELECT zone_id, time_bucket({step}, probe_ts) AS tick_start,
                   sum(speed_kmh) AS s, count(*) AS n
            FROM probes GROUP BY 1, 2
        ), grid AS (
            SELECT z.zone_id, t.tick_start, coalesce(a.s, 0) AS s, coalesce(a.n, 0) AS n
            FROM zones z CROSS JOIN ticks t
            LEFT JOIN agg a ON a.zone_id = z.zone_id AND a.tick_start = t.tick_start
        ), w AS (
            SELECT zone_id, tick_start + {step} AS event_timestamp,
              sum(s) OVER w15 AS s15, sum(n) OVER w15 AS n15,
              sum(s) OVER w60 AS s60, sum(n) OVER w60 AS n60
            FROM grid
            WINDOW w15 AS (PARTITION BY zone_id ORDER BY tick_start ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
                   w60 AS (PARTITION BY zone_id ORDER BY tick_start ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)
        )
        SELECT zone_id, event_timestamp,
               CASE WHEN n15 > 0 THEN s15 / n15 END AS avg_speed_15m,
               CASE WHEN n60 > 0 THEN s60 / n60 END AS avg_speed_60m,
               n15::DOUBLE AS probe_count_15m,
               CASE WHEN n15 > 0 AND n60 > 0 THEN (s15 / n15) / (s60 / n60) END AS speed_trend
        FROM w ORDER BY zone_id, event_timestamp
        """
    ).df()
    zone_demand = con.execute(
        f"""
        WITH agg AS (
            SELECT zone_id, time_bucket({step}, request_ts) AS tick_start, count(*) AS r
            FROM trips GROUP BY 1, 2
        ), grid AS (
            SELECT z.zone_id, t.tick_start, coalesce(a.r, 0) AS r
            FROM zones z CROSS JOIN ticks t
            LEFT JOIN agg a ON a.zone_id = z.zone_id AND a.tick_start = t.tick_start
        ), w AS (
            SELECT zone_id, tick_start + {step} AS event_timestamp,
              sum(r) OVER (PARTITION BY zone_id ORDER BY tick_start ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS r15,
              sum(r) OVER (PARTITION BY zone_id ORDER BY tick_start ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS r60
            FROM grid
        )
        SELECT zone_id, event_timestamp, r15::DOUBLE AS requests_15m, r60::DOUBLE AS requests_60m,
               (4.0 * r15 + 1.0) / (r60 + 1.0) AS demand_index
        FROM w ORDER BY zone_id, event_timestamp
        """
    ).df()
    driver_stats = con.execute(
        f"""
        WITH hours AS (
            SELECT unnest(generate_series(TIMESTAMPTZ '{lo_s}', TIMESTAMPTZ '{hi_s}', INTERVAL 1 HOUR)) AS h
        ), o AS (
            SELECT driver_id, time_bucket(INTERVAL 1 HOUR, offer_ts) AS h, count(*) AS n_off, sum(accepted) AS n_acc
            FROM offers GROUP BY 1, 2
        ), t AS (  -- a trip's outcome is only known at drop-off
            SELECT driver_id, time_bucket(INTERVAL 1 HOUR, dropoff_ts) AS h, count(*) AS n_trip,
                   sum(duration_min / free_flow_min) AS ratio_sum
            FROM trips GROUP BY 1, 2
        ), grid AS (
            SELECT d.driver_id, hs.h, coalesce(o.n_off, 0) AS n_off, coalesce(o.n_acc, 0) AS n_acc,
                   coalesce(t.n_trip, 0) AS n_trip, coalesce(t.ratio_sum, 0) AS ratio_sum
            FROM drivers d CROSS JOIN hours hs
            LEFT JOIN o ON o.driver_id = d.driver_id AND o.h = hs.h
            LEFT JOIN t ON t.driver_id = d.driver_id AND t.h = hs.h
        ), w AS (
            SELECT driver_id, h + INTERVAL 1 HOUR AS event_timestamp,
              sum(n_off) OVER w7 AS off7, sum(n_acc) OVER w7 AS acc7,
              sum(n_trip) OVER w7 AS trip7, sum(ratio_sum) OVER w7 AS ratio7,
              sum(n_trip) OVER (PARTITION BY driver_id ORDER BY h ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS trip24
            FROM grid
            WINDOW w7 AS (PARTITION BY driver_id ORDER BY h ROWS BETWEEN 167 PRECEDING AND CURRENT ROW)
        )
        SELECT driver_id, event_timestamp,
               CASE WHEN off7 > 0 THEN acc7 / off7 END AS acceptance_rate_7d,
               trip24::DOUBLE AS trips_24h,
               CASE WHEN trip7 > 0 THEN ratio7 / trip7 END AS slowness_ratio_7d
        FROM w ORDER BY driver_id, event_timestamp
        """
    ).df()
    con.close()
    out = {"zone_traffic": zone_traffic, "zone_demand": zone_demand, "driver_stats": driver_stats}
    for df in out.values():
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
        df["created"] = df["event_timestamp"]
    return out


def write_feature_tables(feats: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in feats.items():
        p = out_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths[name] = p
    return paths


def hourly_snapshot_tables(tables: dict[str, pd.DataFrame], feats: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The 'analytics warehouse' tables a naive pipeline joins against.

    - zone features: the *last* row of each (zone, clock hour), i.e. the state at the end of the hour
    - driver features: one profile row per driver computed over the whole export (all trips, all offers)
    """
    zt = feats["zone_traffic"].copy()
    zd = feats["zone_demand"].copy()
    for df in (zt, zd):
        # the row stamped HH:00 summarises the previous hour, so bucket by (stamp - 1 tick)
        df["hour"] = (df["event_timestamp"] - pd.Timedelta(seconds=1)).dt.floor("h")
    zt_h = zt.sort_values("event_timestamp").groupby(["zone_id", "hour"], as_index=False).last()
    zd_h = zd.sort_values("event_timestamp").groupby(["zone_id", "hour"], as_index=False).last()
    trips, offers = tables["trips"], tables["offers"]
    end = trips.dropoff_ts.max()
    acc = offers.groupby("driver_id")["accepted"].mean().rename("acceptance_rate_7d")
    t24 = trips[trips.dropoff_ts > end - pd.Timedelta(hours=24)].groupby("driver_id").size().rename("trips_24h")
    slow = (trips.duration_min / trips.free_flow_min).groupby(trips.driver_id).mean().rename("slowness_ratio_7d")
    drv = pd.concat([acc, t24, slow], axis=1).reset_index().rename(columns={"index": "driver_id"})
    drv["trips_24h"] = drv["trips_24h"].fillna(0.0)
    return {
        "zone_traffic_hourly": zt_h[["zone_id", "hour", *ZONE_TRAFFIC_FEATURES]],
        "zone_demand_hourly": zd_h[["zone_id", "hour", *ZONE_DEMAND_FEATURES]],
        "driver_profile": drv[["driver_id", *DRIVER_FEATURES]],
    }
