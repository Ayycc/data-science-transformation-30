"""Dagster code location. ``dagster dev -m air_quality.definitions`` serves it in the UI.

Environment:
- ``AQ_DATA_DIR`` (default ``./data``): landing zone and DuckDB warehouse.
- ``AQ_RESULTS_DIR`` (default ``./results``): where the report asset publishes.
- ``AQ_STORAGE`` = ``local`` (default) or ``s3`` (with ``AQ_S3_BUCKET``; needs boto3 + credentials).
"""

from __future__ import annotations

import os
from pathlib import Path

from dagster import Definitions, in_process_executor
from dagster_duckdb import DuckDBResource

from .assets import ALL_ASSETS
from .config import data_dir, results_dir
from .resources import LocalLandingStorage, LocalReportStore, S3LandingStorage, S3ReportStore
from .sensors import daily_job, daily_schedule, late_data_sensor


def build_resources(
    data: Path | None = None,
    results: Path | None = None,
    storage: str | None = None,
    transient_failures: list[str] | None = None,
) -> dict:
    data = Path(data or data_dir())
    results = Path(results or results_dir())
    storage = storage or os.environ.get("AQ_STORAGE", "local")
    (data / "warehouse").mkdir(parents=True, exist_ok=True)
    if storage == "s3":
        bucket = os.environ["AQ_S3_BUCKET"]
        landing = S3LandingStorage(bucket=bucket, endpoint_url=os.environ.get("AQ_S3_ENDPOINT"))
        reports = S3ReportStore(bucket=bucket, endpoint_url=os.environ.get("AQ_S3_ENDPOINT"))
    elif storage == "local":
        landing = LocalLandingStorage(base_dir=str(data), transient_failures=transient_failures or [])
        reports = LocalReportStore(base_dir=str(results))
    else:
        raise ValueError(f"AQ_STORAGE must be 'local' or 's3', got {storage!r}")
    return {
        "landing": landing,
        "reports": reports,
        "duckdb": DuckDBResource(database=str(data / "warehouse" / "air_quality.duckdb")),
    }


def build_defs(**kwargs) -> Definitions:
    return Definitions(
        assets=ALL_ASSETS,
        jobs=[daily_job],
        schedules=[daily_schedule],
        sensors=[late_data_sensor],
        resources=build_resources(**kwargs),
        # One process: steps share a single DuckDB file, which allows only one writer process at a time.
        executor=in_process_executor,
    )


defs = build_defs()
