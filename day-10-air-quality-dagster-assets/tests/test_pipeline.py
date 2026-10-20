"""Dagster-level tests on the tiny SYNTHETIC preset: backfill, blocking checks, late-data sensor, retries,
headless CLI partition ranges and the storage resource switch."""

from __future__ import annotations

import os
import subprocess
import sys

import duckdb
import pytest
from dagster import DagsterEventType, DagsterInstance, build_sensor_context
from dagster._core.storage.tags import ASSET_PARTITION_RANGE_END_TAG, ASSET_PARTITION_RANGE_START_TAG

from air_quality.config import daily_partitions
from air_quality.definitions import build_defs, build_resources
from air_quality.resources import LocalLandingStorage, S3LandingStorage, S3ReportStore
from air_quality.sensors import late_data_sensor

from .conftest import ROOT, deliver_late

BLOCKED = {"2026-09-12": "raw_completeness", "2026-09-24": "systemic_fault_rate"}


def _setup(data, tmp_path, **kw):
    defs = build_defs(data=data, results=tmp_path / "results", **kw)
    home = tmp_path / "dagster_home"
    home.mkdir(exist_ok=True)
    instance = DagsterInstance.local_temp(tempdir=str(home))
    reg = defs.resolve_assets_def("station_registry").key
    defs.resolve_implicit_global_asset_job_def().execute_in_process(instance=instance, asset_selection=[reg])
    return defs, instance


def _db(data):
    return duckdb.connect(str(data / "warehouse" / "air_quality.duckdb"), read_only=True)


def _mats(result):
    return {e.asset_key.path[-1] for e in result.get_asset_materialization_events()}


def test_definitions_load():
    defs = build_defs()
    type(defs).validate_loadable(defs)
    assert defs.resolve_job_def("daily_air_quality").partitions_def == daily_partitions
    assert len(daily_partitions.get_partition_keys()) == 30


def test_backfill_30_partitions_and_blocking_checks(tiny_data, tmp_path):
    defs, instance = _setup(tiny_data, tmp_path)
    job = defs.resolve_job_def("daily_air_quality")
    outcomes = {}
    for pk in daily_partitions.get_partition_keys():
        r = job.execute_in_process(partition_key=pk, instance=instance, raise_on_error=False)
        failed = {e.check_name for e in r.get_asset_check_evaluations() if not e.passed}
        outcomes[pk] = (r.success, _mats(r), failed)
    assert len(outcomes) == 30
    for pk, (ok, mats, failed) in outcomes.items():
        if pk in BLOCKED:
            assert not ok and BLOCKED[pk] in failed
            assert "alerts" not in mats  # failed blocking check -> downstream alerts not materialised
        else:
            assert ok, (pk, failed)
            assert {"raw_readings", "cleaned_readings", "station_health", "sensor_drift", "daily_aqi", "alerts"} <= mats
    with _db(tiny_data) as c:
        published = {str(d)[:10] for (d,) in c.execute("SELECT date FROM alert_publications").fetchall()}
        n_alert_rows_blocked = c.execute(
            "SELECT count(*) FROM alerts WHERE date IN ('2026-09-12', '2026-09-24')"
        ).fetchone()[0]
    assert published == set(outcomes) - set(BLOCKED)
    assert n_alert_rows_blocked == 0
    # instance-level record of the failed checks
    rec = instance.fetch_materializations(
        __import__("dagster").AssetRecordsFilter(asset_key=__import__("dagster").AssetKey("alerts")), limit=100
    ).records
    assert {r.partition_key for r in rec} == set(outcomes) - set(BLOCKED)


def test_late_data_sensor_backfills_and_unblocks(tiny_data, tmp_path):
    defs, instance = _setup(tiny_data, tmp_path)
    job = defs.resolve_job_def("daily_air_quality")
    for pk in ("2026-09-11", "2026-09-12", "2026-09-20"):
        job.execute_in_process(partition_key=pk, instance=instance, raise_on_error=False)
    landing = build_resources(data=tiny_data, results=tmp_path / "results")["landing"]
    tick0 = late_data_sensor(build_sensor_context(instance=instance, resources={"landing": landing}))
    assert {r.partition_key for r in tick0.run_requests if r.tags["aq/trigger"] == "late_data"} == set()

    late = deliver_late(tiny_data)
    tick1 = late_data_sensor(build_sensor_context(instance=instance, resources={"landing": landing}))
    req = {r.partition_key: r for r in tick1.run_requests}
    late_reqs = {pk for pk, r in req.items() if r.tags["aq/trigger"] == "late_data"}
    assert late_reqs == {"2026-09-12", "2026-09-20"}  # the materialised partitions whose landing changed
    assert {pk for pk, r in req.items() if r.tags["aq/trigger"] == "new_data"} == set(
        daily_partitions.get_partition_keys()
    ) - {
        "2026-09-11",
        "2026-09-12",
        "2026-09-20",
    }
    assert set(late) >= late_reqs
    for pk in sorted(late_reqs):
        r = job.execute_in_process(
            partition_key=pk, instance=instance, raise_on_error=False, tags={"dagster/run_key": req[pk].run_key}
        )
        assert r.success  # 2026-09-12 completeness passes once the buffered readings land
        assert "alerts" in _mats(r)
    tick2 = late_data_sensor(build_sensor_context(instance=instance, resources={"landing": landing}))
    assert not {r.partition_key for r in tick2.run_requests} & late_reqs


def test_retry_policy_recovers_transient_storage_error(tiny_data, tmp_path):
    defs, instance = _setup(tiny_data, tmp_path, transient_failures=["2026-09-02"])
    r = defs.resolve_job_def("daily_air_quality").execute_in_process(
        partition_key="2026-09-02", instance=instance, raise_on_error=False
    )
    restarts = [e for e in r.all_events if e.event_type == DagsterEventType.STEP_RESTARTED]
    assert r.success and len(restarts) == 1 and restarts[0].step_key == "raw_readings"


def test_single_run_partition_range_backfill(tiny_data, tmp_path):
    deliver_late(tiny_data)
    defs, instance = _setup(tiny_data, tmp_path)
    keys = daily_partitions.get_partition_keys()
    clean = [k for k in keys if k < "2026-09-24"]  # 23 partitions without a blocking fault once late data landed
    r = defs.resolve_job_def("daily_air_quality").execute_in_process(
        instance=instance,
        raise_on_error=False,
        tags={ASSET_PARTITION_RANGE_START_TAG: clean[0], ASSET_PARTITION_RANGE_END_TAG: clean[-1]},
    )
    assert r.success
    with _db(tiny_data) as c:
        assert c.execute("SELECT count(DISTINCT date) FROM daily_aqi").fetchone()[0] == len(clean)
        assert c.execute("SELECT count(*) FROM alert_publications").fetchone()[0] == len(clean)
    # full 30-partition range: one bad partition blocks alerts for the whole run (coarse gating)
    r30 = defs.resolve_job_def("daily_air_quality").execute_in_process(
        instance=instance,
        raise_on_error=False,
        tags={ASSET_PARTITION_RANGE_START_TAG: keys[0], ASSET_PARTITION_RANGE_END_TAG: keys[-1]},
    )
    assert not r30.success and "alerts" not in _mats(r30)
    with _db(tiny_data) as c:
        assert c.execute("SELECT count(DISTINCT date) FROM cleaned_readings").fetchone()[0] == 30


def test_cli_materialize_partition_range_headless(tiny_data, tmp_path):
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "AQ_DATA_DIR": str(tiny_data),
        "AQ_RESULTS_DIR": str(tmp_path / "results"),
        "DAGSTER_HOME": str(tmp_path / "home"),
    }
    (tmp_path / "home").mkdir()
    cli = [sys.executable, "-m", "dagster", "asset", "materialize", "-m", "air_quality.definitions"]
    subprocess.run([*cli, "--select", "station_registry"], env=env, check=True, capture_output=True, cwd=ROOT)
    sel = "(group:ingest or group:quality or group:publish) and not key:station_registry"
    p = subprocess.run(
        [*cli, "--select", sel, "--partition-range", "2026-09-01...2026-09-05"],
        env=env,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    with _db(tiny_data) as c:
        dates = [
            str(d)[:10] for (d,) in c.execute("SELECT DISTINCT date FROM alert_publications ORDER BY 1").fetchall()
        ]
    assert dates == ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05"]


def test_report_asset_publishes_files(tiny_data, tmp_path):
    defs, instance = _setup(tiny_data, tmp_path)
    for pk in ("2026-09-03", "2026-09-04", "2026-09-24"):
        defs.resolve_job_def("daily_air_quality").execute_in_process(
            partition_key=pk, instance=instance, raise_on_error=False
        )
    rep = defs.resolve_implicit_global_asset_job_def().execute_in_process(
        instance=instance, asset_selection=[defs.resolve_assets_def("aqi_report").key]
    )
    assert rep.success
    out = tmp_path / "results"
    for f in (
        "aqi_report.md",
        "alerts.csv",
        "daily_summary.csv",
        "region_advisories.csv",
        "figures/aqi_overview.png",
        "figures/aqi_heatmap.html",
    ):
        assert (out / f).exists(), f
    text = (out / "aqi_report.md").read_text()
    assert "2026-09-24 | systemic_fault_rate" in text


def test_storage_resource_switch(tmp_path, monkeypatch):
    local = build_resources(data=tmp_path, results=tmp_path / "r", storage="local")
    assert isinstance(local["landing"], LocalLandingStorage)
    monkeypatch.setenv("AQ_S3_BUCKET", "agency-air-quality")
    cloud = build_resources(data=tmp_path, results=tmp_path / "r", storage="s3")
    assert isinstance(cloud["landing"], S3LandingStorage) and isinstance(cloud["reports"], S3ReportStore)
    assert cloud["landing"].bucket == "agency-air-quality"
    with pytest.raises(ValueError):
        build_resources(data=tmp_path, storage="ftp")
