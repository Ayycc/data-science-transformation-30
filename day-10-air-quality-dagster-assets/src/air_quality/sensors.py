"""Daily job, its schedule, and the late-data sensor that triggers partition backfills."""

from dagster import (
    AssetKey,
    AssetRecordsFilter,
    AssetSelection,
    DefaultSensorStatus,
    RunRequest,
    SensorEvaluationContext,
    SensorResult,
    build_schedule_from_partitioned_job,
    define_asset_job,
    sensor,
)

from .config import daily_partitions
from .resources import LandingStorage

daily_job = define_asset_job(
    "daily_air_quality",
    selection=AssetSelection.groups("ingest", "quality", "publish") - AssetSelection.assets("station_registry"),
    partitions_def=daily_partitions,
    description="All partitioned assets for one (or a range of) daily partition(s).",
)

daily_schedule = build_schedule_from_partitioned_job(daily_job, hour_of_day=3, minute_of_hour=0)

RAW_KEY = AssetKey("raw_readings")


def stale_partitions(instance, landing: LandingStorage) -> list[tuple[str, str, str]]:
    """(partition, landing fingerprint, reason) for partitions whose landing files differ from what was ingested."""
    valid = set(daily_partitions.get_partition_keys())
    out = []
    for pk in landing.list_partitions():
        if pk not in valid:
            continue
        fp = landing.fingerprint(pk)
        recs = instance.fetch_materializations(
            AssetRecordsFilter(asset_key=RAW_KEY, asset_partitions=[pk]), limit=1
        ).records
        if not recs:
            out.append((pk, fp, "new_data"))
            continue
        md = recs[0].asset_materialization.metadata.get("landing_fingerprints")
        ingested = (md.value if md is not None else {}).get(pk)
        if ingested != fp:
            out.append((pk, fp, "late_data"))
    return out


@sensor(job=daily_job, minimum_interval_seconds=300, default_status=DefaultSensorStatus.STOPPED)
def late_data_sensor(context: SensorEvaluationContext, landing: LandingStorage) -> SensorResult:
    """Backfill any partition whose landing zone changed since its last ``raw_readings`` materialisation.

    The ``run_key`` includes the landing fingerprint, so the daemon launches each (partition, data version)
    once; a new late batch changes the fingerprint and triggers a new run.
    """
    reqs = [
        RunRequest(run_key=f"{pk}:{fp}", partition_key=pk, tags={"aq/trigger": reason, "aq/landing_fingerprint": fp})
        for pk, fp, reason in stale_partitions(context.instance, landing)
    ]
    if reqs:
        context.log.info(f"requesting {len(reqs)} partition run(s): {[r.partition_key for r in reqs]}")
    return SensorResult(run_requests=reqs)
