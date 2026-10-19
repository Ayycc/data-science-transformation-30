"""Delta Lake behaviour: incremental MERGE, late events, idempotency, schema, time travel (tiny SYNTHETIC data)."""

from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import functions as F

from clickstream import evaluate
from clickstream import transforms as T
from clickstream.config import TEST
from clickstream.data import write_landing
from clickstream.lakehouse import Lakehouse
from clickstream.pipeline import schema_checks, time_travel_audit

from .test_transforms import EV, ev


@pytest.fixture(scope="module")
def built(spark, tmp_path_factory):
    root = tmp_path_factory.mktemp("lake")
    write_landing(spark, TEST, root / "landing")
    lh = Lakehouse(spark, root / "delta", str(root / "landing" / "deliveries"), TEST.days)
    lh.create_tables()
    versions = {}
    for b in range(1, TEST.days + 1):
        lh.run_batch(b)
        versions[b] = lh.version(lh.sessions)
    return lh, root, versions


def test_rerun_batch_gives_identical_gold(built):
    lh, _, _ = built
    before = lh.gold_checksums()
    bronze_v = lh.version(lh.bronze)
    sessions_v = lh.version(lh.sessions)
    lh.run_batch(2)
    lh.run_batch(TEST.days)
    assert lh.version(lh.bronze) == bronze_v  # txnAppId/txnVersion: replayed appends are skipped
    hist = lh.spark.sql(f"DESCRIBE HISTORY delta.`{lh.sessions}`").collect()
    for h in hist:
        if h["version"] > sessions_v:  # a replay commit, if any, must not change rows
            m = h["operationMetrics"]
            assert all(int(m.get(k, "0")) == 0 for k in ("numTargetRowsInserted", "numTargetRowsUpdated", "numTargetRowsDeleted"))
    assert lh.gold_checksums() == before


def test_incremental_equals_full_recompute(built, spark):
    lh, _, _ = built
    users = spark.read.format("delta").load(lh.silver).select("user_id").distinct()
    prof, sess = lh.recompute_users(users)
    ck = lh.gold_checksums()
    assert T.table_checksum(sess) == ck["gold_sessions"]
    assert T.table_checksum(prof) == ck["gold_user_profiles"]


def test_silver_has_one_row_per_event(built, spark):
    lh, root, _ = built
    silver = spark.read.format("delta").load(lh.silver)
    truth = spark.read.parquet(str(root / "landing" / "deliveries")).filter("truth_delivery = 0")
    assert silver.count() == silver.select("event_id").distinct().count() == truth.count()


def test_sessions_and_bots_match_ground_truth(built, spark):
    lh, root, _ = built
    tu = spark.read.parquet(str(root / "landing" / "truth_users"))
    ts_ = spark.read.parquet(str(root / "landing" / "truth_sessions"))
    sess = spark.read.format("delta").load(lh.sessions)
    rec = evaluate.session_recovery(spark, sess.filter(~F.col("is_bot")), ts_, tu)
    assert rec["exact_recovery_rate"] == 1.0
    bots = evaluate.bot_detection(spark.read.format("delta").load(lh.profiles), tu)
    assert bots["fp"] == 0 and bots["fn"] == 0
    naive = evaluate.naive_session_recovery(spark.read.parquet(str(root / "landing" / "deliveries")), ts_, tu)
    assert naive["exact_recovery_rate"] < rec["exact_recovery_rate"]


def test_schema_enforcement_and_evolution(built):
    lh, _, _ = built
    out = schema_checks(lh)
    assert out["string_timestamp"].startswith("REJECTED")
    assert out["null_event_id"].startswith("REJECTED")
    assert out["unknown_event_type"].startswith("REJECTED")
    assert out["bronze_version_unchanged"]
    assert "app_version" in out["silver_columns"]  # added by the last batch via schema evolution


def test_time_travel_audit(built):
    lh, _, versions = built
    tt = time_travel_audit(lh, versions, "2026-10-01")
    assert tt["sessions_v1"] == tt["unchanged"] + tt["updated"] + tt["removed"]
    assert tt["events_now"] >= tt["events_v1"]  # late events only ever add to day-1 sessions


def test_late_event_merges_two_sessions(spark, tmp_path):
    lh = Lakehouse(spark, tmp_path / "delta", "unused", last_batch=2)
    lh.create_tables()
    d = lambda h, m: datetime(2026, 10, 1, h, m)  # noqa: E731
    b1 = spark.createDataFrame([ev("x1", 1, d(10, 0)), ev("x2", 1, d(10, 50))], EV)
    lh.ingest_bronze(1, b1)
    lh.merge_silver(1)
    lh.merge_gold(1)
    s1 = spark.read.format("delta").load(lh.sessions).collect()
    assert len(s1) == 2
    b2 = spark.createDataFrame([ev("x3", 1, d(10, 25), ingest=datetime(2026, 10, 2, 1, 0)), ev("x1", 1, d(10, 0))], EV)  # late + dup
    lh.ingest_bronze(2, b2)
    lh.merge_silver(2)
    lh.merge_gold(2)
    s2 = spark.read.format("delta").load(lh.sessions).collect()
    assert len(s2) == 1 and s2[0].n_events == 3
    assert s2[0].session_id == min(s1, key=lambda r: r.session_start).session_id  # same start -> same id, updated in place
    m = lh.last_operation_metrics(lh.sessions)
    assert m["numTargetRowsUpdated"] == 1 and m["numTargetRowsDeleted"] == 1
    assert spark.read.format("delta").load(lh.silver).count() == 3
    old = spark.read.format("delta").option("versionAsOf", 1).load(lh.sessions).count()
    assert old == 2
