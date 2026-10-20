"""Software-defined assets: raw -> cleaned -> station health / drift / daily AQI -> alerts -> report.

Every partitioned asset handles one or many daily partitions per run (``context.partition_keys``), so the
same code serves per-partition runs (the sensor, the schedule, a multi-run backfill) and single-run
partition ranges (``dagster asset materialize --partition-range``). Data lives in DuckDB; each asset
replaces exactly its partitions, so re-materialising a partition is idempotent.
"""

import json
from collections.abc import Iterable

import pandas as pd
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetCheckSpec,
    AssetDep,
    AssetExecutionContext,
    BackfillPolicy,
    Backoff,
    MaterializeResult,
    MetadataValue,
    RetryPolicy,
    TimeWindowPartitionMapping,
    asset,
)
from dagster_duckdb import DuckDBResource

from . import report
from .aqi import daily_station_aqi
from .cleaning import clean_readings, detect_drift
from .cleaning import station_health as compute_station_health
from .config import (
    ALERT_AQI,
    DRIFT_WINDOW_DAYS,
    FRESHNESS_MAX_MEDIAN_LATENCY_MIN,
    MAX_SYSTEMIC_FAULT_RATE,
    MIN_AQI_COVERAGE,
    MIN_RAW_COMPLETENESS,
    POLLUTANTS,
    REGION_ADVISORY_SHARE,
    daily_partitions,
)
from .resources import LandingStorage, ReportStore
from .schemas import cleaned_schema, raw_schema, validate

PARTITIONED = dict(partitions_def=daily_partitions, backfill_policy=BackfillPolicy.single_run())


# ---------------------------------------------------------------- DuckDB helpers
def replace_partitions(conn, table: str, df: pd.DataFrame, dates: Iterable[str]) -> None:
    """Delete the given dates from ``table`` and insert ``df`` (creating the table on first use)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")  # stored as TIMESTAMP (midnight), even when empty
    if df.empty:  # an empty object column would otherwise create an INTEGER column
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].astype("string")
    conn.register("_incoming", df)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM _incoming LIMIT 0")
    dates = list(dates)
    if dates:
        conn.execute(f"DELETE FROM {table} WHERE date IN (SELECT UNNEST(?::DATE[]))", [dates])
    conn.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _incoming")
    conn.unregister("_incoming")


def read_dates(conn, table: str, start: str, end: str | None = None) -> pd.DataFrame:
    end = end or start
    return conn.execute(f"SELECT * FROM {table} WHERE date BETWEEN ?::DATE AND ?::DATE", [start, end]).df()


def record_checks(conn, context: AssetExecutionContext, asset_name: str, rows: list[dict]) -> None:
    """Audit table of every per-partition check outcome (Dagster keeps its own history too)."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["asset"] = asset_name
    df["run_id"] = context.run.run_id
    df["evaluated_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    conn.register("_chk", df)
    conn.execute("CREATE TABLE IF NOT EXISTS check_results AS SELECT * FROM _chk LIMIT 0")
    conn.execute(
        "DELETE FROM check_results WHERE asset = ? AND EXISTS (SELECT 1 FROM _chk c WHERE c.date = check_results.date AND c.check_name = check_results.check_name)",
        [asset_name],
    )
    conn.execute("INSERT INTO check_results BY NAME SELECT * FROM _chk")
    conn.unregister("_chk")


def _summarise(rows: list[dict], name: str) -> tuple[bool, dict]:
    sub = [r for r in rows if r["check_name"] == name]
    failed = [r["date"] for r in sub if not r["passed"]]
    meta = {
        "partitions_evaluated": len(sub),
        "failed_partitions": MetadataValue.json(failed),
        "values": MetadataValue.json({r["date"]: r["value"] for r in sub}),
        "threshold": sub[0]["threshold"] if sub else None,
    }
    return not failed, meta


def _day_bounds(pk: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(pk)
    return start, start + pd.Timedelta(days=1)


# ---------------------------------------------------------------- reference
@asset(group_name="ingest", kinds={"duckdb"}, description="Station registry with known calibration offsets.")
def station_registry(landing: LandingStorage, duckdb: DuckDBResource) -> MaterializeResult:
    reg = landing.read_registry()
    with duckdb.get_connection() as conn:
        conn.register("_reg", reg)
        conn.execute("CREATE OR REPLACE TABLE stations AS SELECT * FROM _reg")
    return MaterializeResult(
        metadata={
            "stations": len(reg),
            "regions": int(reg["region"].nunique()),
            "with_calibration_offset": int((reg["cal_offset_pm25"] != 0).sum()),
        }
    )


def _table_exists(conn, name: str) -> bool:
    return bool(
        conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]).fetchone()[0]
    )


def _stations(conn) -> pd.DataFrame:
    return conn.execute("SELECT * FROM stations ORDER BY station_id").df()


# ---------------------------------------------------------------- raw
@asset(
    **PARTITIONED,
    group_name="ingest",
    kinds={"duckdb", "pandas"},
    deps=[station_registry],
    retry_policy=RetryPolicy(max_retries=3, delay=0.2, backoff=Backoff.EXPONENTIAL),
    check_specs=[
        AssetCheckSpec(
            "raw_schema", asset="raw_readings", blocking=True, description="Pandera contract on every landed batch"
        ),
        AssetCheckSpec(
            "raw_completeness",
            asset="raw_readings",
            blocking=True,
            description=f"Distinct station-hours >= {MIN_RAW_COMPLETENESS:.0%} of registry x 24",
        ),
        AssetCheckSpec(
            "raw_freshness",
            asset="raw_readings",
            blocking=False,
            description=f"Data reaches 23:00 and median ingest latency <= {FRESHNESS_MAX_MEDIAN_LATENCY_MIN:.0f} min",
        ),
    ],
    description="Hourly PM2.5/NO2/O3 readings exactly as they landed (all batches, duplicates kept).",
)
def raw_readings(context: AssetExecutionContext, landing: LandingStorage, duckdb: DuckDBResource) -> MaterializeResult:
    rows: list[dict] = []
    fingerprints: dict[str, str] = {}
    totals = {"rows": 0, "batches": 0}
    frames = []
    with duckdb.get_connection() as conn:
        n_reg = len(_stations(conn))
    for pk in context.partition_keys:
        batches = landing.list_batches(pk)
        fingerprints[pk] = landing.fingerprint(pk)
        parts, schema_ok, schema_fail, examples = [], True, 0, []
        for name, _ in batches:
            b = landing.read_batch(pk, name)
            ok, n_fail, ex = validate(raw_schema, b)
            schema_ok &= ok
            schema_fail += n_fail
            examples += ex
            parts.append(b.assign(batch=name))
        cols = ["station_id", "ts", "pm25", "no2", "o3", "landed_at", "batch"]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
        start, end = _day_bounds(pk)
        df = df[(df["ts"] >= start) & (df["ts"] < end)] if len(df) else df
        df.insert(0, "date", start.date())
        frames.append(df)
        totals["rows"] += len(df)
        totals["batches"] += len(batches)

        station_hours = int(df[["station_id", "ts"]].drop_duplicates().shape[0]) if len(df) else 0
        completeness = station_hours / (n_reg * 24) if n_reg else 0.0
        if len(df):
            latest_hour = int(pd.Timestamp(df["ts"].max()).hour)
            latency = float(((df["landed_at"] - df["ts"]).dt.total_seconds() / 60).median())
        else:
            latest_hour, latency = -1, float("inf")
        fresh = latest_hour == 23 and latency <= FRESHNESS_MAX_MEDIAN_LATENCY_MIN
        rows += [
            dict(
                date=pk,
                check_name="raw_schema",
                passed=bool(schema_ok and len(batches) > 0),
                value=float(schema_fail),
                threshold=0.0,
            ),
            dict(
                date=pk,
                check_name="raw_completeness",
                passed=completeness >= MIN_RAW_COMPLETENESS,
                value=round(completeness, 4),
                threshold=MIN_RAW_COMPLETENESS,
            ),
            dict(
                date=pk,
                check_name="raw_freshness",
                passed=bool(fresh),
                value=round(latency, 1),
                threshold=FRESHNESS_MAX_MEDIAN_LATENCY_MIN,
            ),
        ]
        context.log.info(f"{pk}: {len(batches)} batch(es), {len(df)} rows, completeness {completeness:.3f}")

    with duckdb.get_connection() as conn:
        replace_partitions(conn, "raw_readings", pd.concat(frames, ignore_index=True), context.partition_keys)
        record_checks(conn, context, "raw_readings", rows)

    checks = []
    for name, sev in (
        ("raw_schema", AssetCheckSeverity.ERROR),
        ("raw_completeness", AssetCheckSeverity.ERROR),
        ("raw_freshness", AssetCheckSeverity.WARN),
    ):
        passed, meta = _summarise(rows, name)
        checks.append(AssetCheckResult(check_name=name, passed=passed, severity=sev, metadata=meta))
    return MaterializeResult(
        metadata={
            "rows": totals["rows"],
            "batches": totals["batches"],
            "landing_fingerprints": MetadataValue.json(fingerprints),
            "completeness": MetadataValue.json(
                {r["date"]: r["value"] for r in rows if r["check_name"] == "raw_completeness"}
            ),
        },
        check_results=checks,
    )


# ---------------------------------------------------------------- cleaned
@asset(
    **PARTITIONED,
    group_name="quality",
    kinds={"duckdb", "pandas"},
    deps=[raw_readings, station_registry],
    check_specs=[
        AssetCheckSpec(
            "value_ranges",
            asset="cleaned_readings",
            blocking=True,
            description="Pandera: values within physical bounds, unique (station, ts)",
        ),
        AssetCheckSpec(
            "systemic_fault_rate",
            asset="cleaned_readings",
            blocking=True,
            description=f"Raw values rejected as out of range <= {MAX_SYSTEMIC_FAULT_RATE:.0%} (a higher rate means a feed-wide fault)",
        ),
    ],
    description="Deduplicated readings with out-of-range, flatline and spike values nulled and calibration offsets applied.",
)
def cleaned_readings(context: AssetExecutionContext, duckdb: DuckDBResource) -> MaterializeResult:
    rows: list[dict] = []
    frames = []
    agg: dict[str, int] = {}
    with duckdb.get_connection() as conn:
        stations = _stations(conn)
        for pk in context.partition_keys:
            raw = read_dates(conn, "raw_readings", pk).drop(columns=["date", "batch"])
            clean, stats = clean_readings(raw, stations)
            for k, v in stats.items():
                agg[k] = agg.get(k, 0) + v
            ok, n_fail, _ = validate(cleaned_schema, clean)
            fault_rate = stats["out_of_range_values"] / max(1, stats["raw_values"])
            rows += [
                dict(date=pk, check_name="value_ranges", passed=ok, value=float(n_fail), threshold=0.0),
                dict(
                    date=pk,
                    check_name="systemic_fault_rate",
                    passed=fault_rate <= MAX_SYSTEMIC_FAULT_RATE,
                    value=round(fault_rate, 4),
                    threshold=MAX_SYSTEMIC_FAULT_RATE,
                ),
            ]
            clean.insert(0, "date", pd.Timestamp(pk).date())
            frames.append(clean)
        replace_partitions(conn, "cleaned_readings", pd.concat(frames, ignore_index=True), context.partition_keys)
        record_checks(conn, context, "cleaned_readings", rows)
    checks = []
    for name in ("value_ranges", "systemic_fault_rate"):
        passed, meta = _summarise(rows, name)
        checks.append(
            AssetCheckResult(check_name=name, passed=passed, severity=AssetCheckSeverity.ERROR, metadata=meta)
        )
    return MaterializeResult(metadata={k: int(v) for k, v in agg.items()}, check_results=checks)


# ---------------------------------------------------------------- station health and drift
@asset(
    **PARTITIONED,
    group_name="quality",
    kinds={"duckdb", "pandas"},
    deps=[cleaned_readings],
    description="Daily uptime and fault counts per station.",
)
def station_health(context: AssetExecutionContext, duckdb: DuckDBResource) -> MaterializeResult:
    frames = []
    with duckdb.get_connection() as conn:
        stations = _stations(conn)
        for pk in context.partition_keys:
            clean = read_dates(conn, "cleaned_readings", pk)
            h = compute_station_health(clean, stations, pd.Timestamp(pk).date())
            frames.append(h)
        df = pd.concat(frames, ignore_index=True)
        replace_partitions(conn, "station_health", df, context.partition_keys)
    counts = df["status"].value_counts().to_dict()
    return MaterializeResult(
        metadata={"station_days": len(df), "status_counts": MetadataValue.json({k: int(v) for k, v in counts.items()})}
    )


@asset(
    **PARTITIONED,
    group_name="quality",
    kinds={"duckdb", "numpy"},
    deps=[
        AssetDep(
            cleaned_readings,
            partition_mapping=TimeWindowPartitionMapping(
                start_offset=-(DRIFT_WINDOW_DAYS - 1), allow_nonexistent_upstream_partitions=True
            ),
        )
    ],
    description=f"PM2.5 drift rate per sensor from a {DRIFT_WINDOW_DAYS}-day trailing regression against the regional median.",
)
def sensor_drift(context: AssetExecutionContext, duckdb: DuckDBResource) -> MaterializeResult:
    frames = []
    with duckdb.get_connection() as conn:
        stations = _stations(conn)
        for pk in context.partition_keys:
            start = (pd.Timestamp(pk) - pd.Timedelta(days=DRIFT_WINDOW_DAYS - 1)).strftime("%Y-%m-%d")
            window = read_dates(conn, "cleaned_readings", start, pk)
            res = detect_drift(window, stations, pd.Timestamp(pk).date())
            # Latch: a sensor stays suspect from its first detection until it is recalibrated, because a
            # drifting sensor that has drifted to 0 (and been nulled as a flatline) no longer shows a slope.
            res["detected_today"] = res["drift_suspect"]
            res["first_flagged"] = pd.to_datetime(res["date"]).where(res["detected_today"])
            if _table_exists(conn, "sensor_drift"):
                prev_day = (pd.Timestamp(pk) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                prev = read_dates(conn, "sensor_drift", prev_day)[["station_id", "drift_suspect", "first_flagged"]]
                prev = prev[prev["drift_suspect"]].set_index("station_id")
                carried = res["station_id"].isin(prev.index)
                res.loc[carried, "first_flagged"] = res.loc[carried, "station_id"].map(prev["first_flagged"])
                res["drift_suspect"] = res["detected_today"] | carried
            replace_partitions(conn, "sensor_drift", res, [pk])  # written per day: the next day reads it
            frames.append(res)
        df = pd.concat(frames, ignore_index=True)
    flagged = df[df["drift_suspect"]]
    return MaterializeResult(
        metadata={
            "drift_suspects": int(len(flagged)),
            "suspects": MetadataValue.json(sorted(flagged["station_id"].unique().tolist())),
        }
    )


# ---------------------------------------------------------------- AQI
@asset(
    **PARTITIONED,
    group_name="publish",
    kinds={"duckdb", "pandas"},
    deps=[cleaned_readings],
    check_specs=[
        AssetCheckSpec(
            "aqi_coverage",
            asset="daily_aqi",
            blocking=True,
            description=f"Registry stations with a valid PM2.5 sub-index >= {MIN_AQI_COVERAGE:.0%}",
        )
    ],
    description="US EPA daily AQI per station (PM2.5 24-h mean, NO2 max 1-h, O3 max 8-h).",
)
def daily_aqi(context: AssetExecutionContext, duckdb: DuckDBResource) -> MaterializeResult:
    rows, frames = [], []
    with duckdb.get_connection() as conn:
        n_reg = len(_stations(conn))
        for pk in context.partition_keys:
            clean = read_dates(conn, "cleaned_readings", pk)
            a = daily_station_aqi(clean[["station_id", "ts", *POLLUTANTS]])
            a["date"] = pd.to_datetime(a["date"]).dt.date
            frames.append(a)
            coverage = float(a["aqi_pm25"].notna().sum()) / n_reg
            rows.append(
                dict(
                    date=pk,
                    check_name="aqi_coverage",
                    passed=coverage >= MIN_AQI_COVERAGE,
                    value=round(coverage, 4),
                    threshold=MIN_AQI_COVERAGE,
                )
            )
        df = pd.concat(frames, ignore_index=True)
        replace_partitions(conn, "daily_aqi", df, context.partition_keys)
        record_checks(conn, context, "daily_aqi", rows)
    passed, meta = _summarise(rows, "aqi_coverage")
    return MaterializeResult(
        metadata={"station_days": len(df), "median_aqi": float(df["aqi"].median()), "max_aqi": float(df["aqi"].max())},
        check_results=[
            AssetCheckResult(check_name="aqi_coverage", passed=passed, severity=AssetCheckSeverity.ERROR, metadata=meta)
        ],
    )


# ---------------------------------------------------------------- alerts
@asset(
    **PARTITIONED,
    group_name="publish",
    kinds={"duckdb", "pandas"},
    deps=[daily_aqi, station_health, sensor_drift],
    description=f"Station alerts (AQI > {ALERT_AQI}) from healthy, non-drifting stations, plus regional advisories.",
)
def alerts(context: AssetExecutionContext, duckdb: DuckDBResource) -> MaterializeResult:
    alert_frames, adv_frames = [], []
    with duckdb.get_connection() as conn:
        stations = _stations(conn)[["station_id", "region", "station_type"]]
        for pk in context.partition_keys:
            a = read_dates(conn, "daily_aqi", pk)
            h = read_dates(conn, "station_health", pk)[["station_id", "status"]]
            d = read_dates(conn, "sensor_drift", pk)[["station_id", "drift_suspect"]]
            m = (
                a.merge(stations, on="station_id")
                .merge(h, on="station_id", how="left")
                .merge(d, on="station_id", how="left")
            )
            m["drift_suspect"] = m["drift_suspect"].astype("boolean").fillna(False).astype(bool)
            eligible = m[m["aqi"].notna() & (m["status"] != "offline") & ~m["drift_suspect"]]
            al = eligible[eligible["aqi"] > ALERT_AQI].copy()
            al = al[["date", "station_id", "region", "station_type", "aqi", "category", "dominant_pollutant", "status"]]
            alert_frames.append(al.sort_values(["aqi", "station_id"], ascending=[False, True]))
            reg = eligible.groupby("region").agg(
                valid_stations=("station_id", "count"),
                alert_stations=("aqi", lambda x: int((x > ALERT_AQI).sum())),
                max_aqi=("aqi", "max"),
            )
            reg = reg.reset_index()
            reg["alert_share"] = (reg["alert_stations"] / reg["valid_stations"]).round(4)
            reg = reg[reg["alert_share"] >= REGION_ADVISORY_SHARE].copy()
            reg.insert(0, "date", pd.Timestamp(pk).date())
            adv_frames.append(reg)
        al_df = pd.concat(alert_frames, ignore_index=True)
        adv_df = pd.concat(adv_frames, ignore_index=True)
        replace_partitions(conn, "alerts", al_df, context.partition_keys)
        replace_partitions(conn, "region_advisories", adv_df, context.partition_keys)
        pub = pd.DataFrame(
            {
                "date": [pd.Timestamp(pk).date() for pk in context.partition_keys],
                "station_alerts": [
                    int((pd.to_datetime(al_df["date"]) == pd.Timestamp(pk)).sum()) for pk in context.partition_keys
                ],
                "run_id": context.run.run_id,
                "published_at": pd.Timestamp.now(tz="UTC").tz_localize(None),
            }
        )
        replace_partitions(conn, "alert_publications", pub, context.partition_keys)
    preview = al_df.head(10)[["date", "station_id", "region", "aqi", "category", "dominant_pollutant"]]
    return MaterializeResult(
        metadata={
            "station_alerts": int(len(al_df)),
            "regional_advisories": int(len(adv_df)),
            "preview": MetadataValue.md(report.md_table(preview) if len(preview) else "No alerts"),
        }
    )


# ---------------------------------------------------------------- report (unpartitioned, reads all partitions)
@asset(
    group_name="reporting",
    kinds={"duckdb", "plotly"},
    deps=[alerts, daily_aqi],
    description="Daily AQI report, alert list and figures published to the report store.",
)
def aqi_report(context: AssetExecutionContext, duckdb: DuckDBResource, reports: ReportStore) -> MaterializeResult:
    with duckdb.get_connection() as conn:
        written = report.publish(conn, reports)
    context.log.info(json.dumps(written))
    return MaterializeResult(metadata={"files": MetadataValue.json(written)})


ALL_ASSETS = [
    station_registry,
    raw_readings,
    cleaned_readings,
    station_health,
    sensor_drift,
    daily_aqi,
    alerts,
    aqi_report,
]
