"""End-to-end run behind ``make run`` (SYNTHETIC data, CPU only, a few minutes).

1. (Re)generate the synthetic month and reset the warehouse and the Dagster instance.
2. Backfill all 30 daily partitions, one run per partition (a transient storage error is injected
   on 2026-09-03 to exercise the RetryPolicy).
3. Deliver the late batches, tick the late-data sensor, execute the runs it requests, tick again.
4. Run a single-run backfill of the same 30 partitions on a copy of the data (the
   ``--partition-range`` code path) to compare both backfill modes.
5. Publish the report asset, score alerts and drift against ground truth, write results/metrics.json.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import duckdb as duckdb_lib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dagster import (  # noqa: E402
    DagsterEventType,
    DagsterInstance,
    build_sensor_context,
)
from dagster._core.storage.tags import ASSET_PARTITION_RANGE_END_TAG, ASSET_PARTITION_RANGE_START_TAG  # noqa: E402

from air_quality import evaluate  # noqa: E402
from air_quality.config import FULL, daily_partitions, data_dir, results_dir  # noqa: E402
from air_quality.definitions import build_defs, build_resources  # noqa: E402
from air_quality.generate import write_dataset  # noqa: E402
from air_quality.sensors import late_data_sensor  # noqa: E402

TRANSIENT_PARTITION = "2026-09-03"


def run_record(result, pk: str, phase: str, seconds: float) -> dict:
    evals = result.get_asset_check_evaluations()
    mats = sorted({e.asset_key.path[-1] for e in result.get_asset_materialization_events()})
    restarts = [e for e in result.all_events if e.event_type == DagsterEventType.STEP_RESTARTED]
    return {
        "phase": phase,
        "partition": pk,
        "run_id": result.run_id,
        "success": bool(result.success),
        "seconds": round(seconds, 2),
        "materialized": ",".join(mats),
        "alerts_materialized": "alerts" in mats,
        "failed_checks": ",".join(sorted(e.check_name for e in evals if not e.passed)),
        "checks_evaluated": len(evals),
        "checks_passed": sum(e.passed for e in evals),
        "step_retries": len(restarts),
    }


def _check_value(evaluation, pk: str):
    md = evaluation.metadata.get("values")
    return None if md is None else md.value.get(pk)


def check_rows(result, pk: str, phase: str) -> list[dict]:
    return [
        {
            "phase": phase,
            "partition": pk,
            "asset": e.asset_key.path[-1],
            "check": e.check_name,
            "passed": bool(e.passed),
            "severity": e.severity.value,
            "value": _check_value(e, pk),
        }
        for e in result.get_asset_check_evaluations()
    ]


def main() -> None:
    t_all = time.perf_counter()
    data, results = data_dir(), results_dir()
    for sub in (
        "landing",
        "late_arrivals",
        "warehouse",
        "dagster_home",
        "truth",
        "reference",
        ".transient",
        "range_demo",
    ):
        shutil.rmtree(data / sub, ignore_errors=True)
    gen = write_dataset(FULL, data)
    print(
        f"generated SYNTHETIC data: {gen['rows_on_time_incl_duplicates']:,} on-time rows, {gen['rows_late']:,} late rows"
    )

    (data / "dagster_home").mkdir(parents=True, exist_ok=True)
    instance = DagsterInstance.local_temp(tempdir=str(data / "dagster_home"))
    defs = build_defs(data=data, results=results, transient_failures=[TRANSIENT_PARTITION])
    job = defs.resolve_job_def("daily_air_quality")
    registry_job = defs.resolve_implicit_global_asset_job_def()

    registry_job.execute_in_process(
        instance=instance, asset_selection=[defs.resolve_assets_def("station_registry").key]
    )
    runs: list[dict] = []
    checks: list[dict] = []

    # ---- phase 1: backfill 30 partitions, one run each ---------------------------------------------
    keys = daily_partitions.get_partition_keys()
    t0 = time.perf_counter()
    for pk in keys:
        t = time.perf_counter()
        r = job.execute_in_process(
            partition_key=pk, instance=instance, raise_on_error=False, tags={"aq/trigger": "backfill"}
        )
        runs.append(run_record(r, pk, "backfill", time.perf_counter() - t))
        checks += check_rows(r, pk, "backfill")
        print(
            f"backfill {pk}: success={r.success} failed_checks={runs[-1]['failed_checks'] or '-'} retries={runs[-1]['step_retries']}"
        )
    backfill_seconds = time.perf_counter() - t0

    db = data / "warehouse" / "air_quality.duckdb"
    with duckdb_lib.connect(str(db), read_only=True) as conn:
        alerts_before = (
            conn.execute("SELECT strftime(date, '%Y-%m-%d') AS d, count(*) AS n FROM alerts GROUP BY 1")
            .df()
            .set_index("d")["n"]
            .to_dict()
        )

    # ---- phase 2: late batches land; the sensor backfills the affected partitions -------------------
    resources = build_resources(data=data, results=results)
    late_files = sorted((data / "late_arrivals").glob("*/batch_*.parquet"))
    for f in late_files:
        dest = data / "landing" / f.parent.name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / f.name)
    tick1 = late_data_sensor(build_sensor_context(instance=instance, resources={"landing": resources["landing"]}))
    requested = [(rr.partition_key, rr.run_key, rr.tags.get("aq/trigger")) for rr in tick1.run_requests]
    print(f"sensor tick 1 requested: {requested}")
    t0 = time.perf_counter()
    for pk, run_key, trigger in requested:
        t = time.perf_counter()
        r = job.execute_in_process(
            partition_key=pk,
            instance=instance,
            raise_on_error=False,
            tags={"aq/trigger": trigger, "dagster/run_key": run_key},
        )
        runs.append(run_record(r, pk, "sensor", time.perf_counter() - t))
        checks += check_rows(r, pk, "sensor")
        print(f"sensor run {pk}: success={r.success} failed_checks={runs[-1]['failed_checks'] or '-'}")
    sensor_seconds = time.perf_counter() - t0
    tick2 = late_data_sensor(build_sensor_context(instance=instance, resources={"landing": resources["landing"]}))
    print(f"sensor tick 2 requested: {len(tick2.run_requests)} run(s)")

    with duckdb_lib.connect(str(db), read_only=True) as conn:
        alerts_after = (
            conn.execute("SELECT strftime(date, '%Y-%m-%d') AS d, count(*) AS n FROM alerts GROUP BY 1")
            .df()
            .set_index("d")["n"]
            .to_dict()
        )

    # ---- report asset ------------------------------------------------------------------------------
    rep = registry_job.execute_in_process(
        instance=instance, asset_selection=[defs.resolve_assets_def("aqi_report").key]
    )
    assert rep.success

    # ---- phase 3: single-run backfill of the same 30 partitions on a copy (partition-range path) -----
    demo = data / "range_demo"
    shutil.copytree(data / "landing", demo / "landing")
    shutil.copytree(data / "reference", demo / "reference")
    demo_defs = build_defs(data=demo, results=demo / "results")
    demo_inst = DagsterInstance.ephemeral()
    demo_defs.resolve_implicit_global_asset_job_def().execute_in_process(
        instance=demo_inst, asset_selection=[demo_defs.resolve_assets_def("station_registry").key]
    )
    t = time.perf_counter()
    rr = demo_defs.resolve_job_def("daily_air_quality").execute_in_process(
        instance=demo_inst,
        raise_on_error=False,
        tags={ASSET_PARTITION_RANGE_START_TAG: keys[0], ASSET_PARTITION_RANGE_END_TAG: keys[-1]},
    )
    range_seconds = time.perf_counter() - t
    range_rec = run_record(rr, f"{keys[0]}...{keys[-1]}", "single_run_range", range_seconds)
    with duckdb_lib.connect(str(demo / "warehouse" / "air_quality.duckdb"), read_only=True) as c:
        tbls = set(c.execute("SELECT table_name FROM information_schema.tables").df()["table_name"])
        range_parts = {
            t: (c.execute(f"SELECT count(DISTINCT date) FROM {t}").fetchone()[0] if t in tbls else 0)
            for t in ("raw_readings", "cleaned_readings", "daily_aqi", "alerts")
        }
        range_alert_parts = (
            c.execute("SELECT count(*) FROM alert_publications").fetchone()[0] if "alert_publications" in tbls else 0
        )
    print(f"single-run range: success={rr.success} failed={range_rec['failed_checks']} seconds={range_seconds:.1f}")

    # ---- evaluation ----------------------------------------------------------------------------------
    with duckdb_lib.connect(str(db), read_only=True) as conn:
        summary = pd.read_csv(results / "daily_summary.csv")
        published = summary.loc[summary["alerts_published"], "date"].tolist()
        blocked = summary.loc[~summary["alerts_published"], ["date", "failed_blocking_checks"]].to_dict("records")
        aq = evaluate.alert_quality(conn, data, published)
        dq = evaluate.drift_quality(conn, data)
        n_adv = conn.execute("SELECT count(*) FROM region_advisories").fetchone()[0]
        table_rows = {
            t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("raw_readings", "cleaned_readings", "daily_aqi", "station_health", "sensor_drift", "alerts")
        }
        health = (
            conn.execute("SELECT status, count(*) AS n FROM station_health GROUP BY 1")
            .df()
            .set_index("status")["n"]
            .to_dict()
        )
        cleaned_stats = (
            conn.execute(
                "SELECT sum((pm25_flag='flatline')::int + (no2_flag='flatline')::int + (o3_flag='flatline')::int) AS flatline, "
                "sum((pm25_flag='spike')::int + (no2_flag='spike')::int + (o3_flag='spike')::int) AS spike, "
                "sum((pm25_flag='out_of_range')::int + (no2_flag='out_of_range')::int + (o3_flag='out_of_range')::int) AS out_of_range FROM cleaned_readings"
            )
            .df()
            .iloc[0]
            .to_dict()
        )
        dup_removed = int(
            conn.execute("SELECT count(*) FROM raw_readings").fetchone()[0]
            - conn.execute("SELECT count(*) FROM cleaned_readings").fetchone()[0]
        )

    runs_df = pd.DataFrame(runs)
    checks_df = pd.DataFrame(checks)
    runs_df.to_csv(results / "run_log.csv", index=False)
    checks_df.to_csv(results / "check_evaluations.csv", index=False)
    late_parts = sorted({f.parent.name for f in late_files})
    bf = runs_df[runs_df["phase"] == "backfill"]
    first_blocked = sorted(bf.loc[~bf["alerts_materialized"], "partition"].tolist())
    late_impact = {
        d: {
            "alerts_before": None if d in first_blocked else int(alerts_before.get(d, 0)),
            "alerts_after": None if d not in published else int(alerts_after.get(d, 0)),
        }
        for d in late_parts
    }
    chk_summary = (
        checks_df.groupby(["asset", "check"])["passed"]
        .agg(evaluations="count", passed="sum")
        .assign(failed=lambda d: d["evaluations"] - d["passed"])
        .reset_index()
    )
    metrics = {
        "data_source": "SYNTHETIC (seeded generator; OpenAQ API not used: blocked by the build network, see README)",
        "generator": gen,
        "orchestration": {
            "dagster_version": __import__("dagster").__version__,
            "partitions": len(keys),
            "backfill_runs": int(len(bf)),
            "backfill_runs_succeeded": int(bf["success"].sum()),
            "backfill_runs_blocked": int((~bf["success"]).sum()),
            "backfill_partitions_with_alerts_published": int(bf["alerts_materialized"].sum()),
            "backfill_blocked_partitions": first_blocked,
            "backfill_failed_checks": {
                r["partition"]: r["failed_checks"] for _, r in bf[bf["failed_checks"] != ""].iterrows()
            },
            "backfill_seconds": round(backfill_seconds, 1),
            "backfill_seconds_per_partition_median": float(bf["seconds"].median()),
            "transient_failure_partition": TRANSIENT_PARTITION,
            "step_retries_total": int(runs_df["step_retries"].sum()),
            "transient_partition_run_success": bool(bf.loc[bf["partition"] == TRANSIENT_PARTITION, "success"].iloc[0]),
            "late_files_delivered": len(late_files),
            "sensor_tick1_requests": [p for p, _, _ in requested],
            "sensor_tick1_reasons": sorted({t for _, _, t in requested}),
            "sensor_runs_succeeded": int(runs_df[runs_df["phase"] == "sensor"]["success"].sum()),
            "sensor_seconds": round(sensor_seconds, 1),
            "sensor_tick2_requests": len(tick2.run_requests),
            "late_data_alert_impact": late_impact,
            "final_published_partitions": len(published),
            "final_blocked_partitions": blocked,
            "single_run_range_backfill": {
                "partitions": len(keys),
                "success": bool(rr.success),
                "seconds": round(range_seconds, 1),
                "failed_checks": range_rec["failed_checks"],
                "partitions_materialized": {k: int(v) for k, v in range_parts.items()},
                "partitions_with_alerts_published": int(range_alert_parts),
            },
            "asset_checks": chk_summary.to_dict("records"),
            "check_evaluations_total": int(len(checks_df)),
            "check_evaluations_failed": int((~checks_df["passed"]).sum()),
            "failed_check_evaluations": checks_df[~checks_df["passed"]].to_dict("records"),
        },
        "warehouse_rows": {k: int(v) for k, v in table_rows.items()},
        "cleaning": {"duplicates_removed": dup_removed, **{k: int(v) for k, v in cleaned_stats.items()}},
        "station_health_station_days": {k: int(v) for k, v in health.items()},
        "alerts": {
            "station_alerts_published": int(sum(alerts_after.values())),
            "regional_advisories": int(n_adv),
            **aq,
        },
        "drift_detection": dq,
        "runtime_seconds_total": round(time.perf_counter() - t_all, 1),
    }
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: metrics[k] for k in ("alerts", "drift_detection")}, indent=1, default=str))
    print(f"done in {metrics['runtime_seconds_total']} s")


if __name__ == "__main__":
    main()
