from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

from clickstream import transforms as T
from clickstream.config import TEST
from clickstream.generate import raw_deliveries, true_sessions

ROOT = Path(__file__).resolve().parents[1]


def test_generator_is_deterministic(spark):
    a = T.table_checksum(raw_deliveries(spark, TEST))
    b = T.table_checksum(raw_deliveries(spark, TEST).repartition(7))
    assert a == b


def test_true_sessions_respect_gap_rule(spark):
    s = true_sessions(spark, TEST).withColumn("end_s", F.col("start_s") + F.col("dur_s"))
    w = Window.partitionBy("user_id").orderBy("start_s")
    gaps = s.withColumn("gap", F.col("start_s") - F.lag("end_s").over(w)).filter(F.col("gap").isNotNull())
    assert gaps.agg(F.min("gap")).collect()[0][0] > 30 * 60
    assert s.agg(F.max("dur_s")).collect()[0][0] < 30 * 60


def test_planted_defects_present(spark):
    d = raw_deliveries(spark, TEST)
    assert d.filter("truth_delivery = 1").count() > 0  # duplicates
    assert d.filter(F.to_date("truth_ts") < F.to_date("ingest_ts")).count() > 0  # late
    assert d.filter(F.col("client_ts") != F.col("truth_ts")).count() > 0  # clock skew
    assert d.filter("truth_is_bot").count() > 0


def test_committed_sample_matches_generator(spark):
    path = ROOT / "data" / "sample" / "deliveries_sample.csv.gz"
    assert path.stat().st_size <= 1_000_000
    s = pd.read_csv(path)
    g = raw_deliveries(spark, TEST).orderBy("batch_id", "ingest_ts", "event_id").toPandas()
    assert len(s) == len(g)
    assert list(s["event_id"]) == list(g["event_id"])
