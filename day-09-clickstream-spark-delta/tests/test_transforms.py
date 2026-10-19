"""chispa unit tests for the pure transformations."""

from __future__ import annotations

from datetime import datetime, timedelta

from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType, TimestampType

from clickstream import transforms as T

EV = StructType(
    [
        StructField("event_id", StringType()),
        StructField("user_id", LongType()),
        StructField("article_id", LongType()),
        StructField("event_type", StringType()),
        StructField("client_ts", TimestampType()),
        StructField("ingest_ts", TimestampType()),
        StructField("device", StringType()),
        StructField("user_agent", StringType()),
        StructField("scroll_depth", IntegerType()),
    ]
)
UA = "Mozilla/5.0 (Windows NT 10.0) Chrome/130.0"


def ts(h, m=0, s=0, d=1):
    return datetime(2026, 10, d, h, m, s)


def ev(eid, uid, t, art=1, typ="page_view", ingest=None, ua=UA, depth=None):
    return (eid, uid, art, typ, t, ingest or t, "desktop", ua, depth)


def test_dedup_keeps_earliest_delivery(spark):
    df = spark.createDataFrame([ev("a", 1, ts(10), ingest=ts(10, 0, 5)), ev("a", 1, ts(10), ingest=ts(12)), ev("b", 1, ts(11))], EV)
    exp = spark.createDataFrame([ev("a", 1, ts(10), ingest=ts(10, 0, 5)), ev("b", 1, ts(11))], EV)
    assert_df_equality(T.dedup_deliveries(df), exp, ignore_row_order=True)


def test_clock_offset_whole_hours_and_late_events_ignored(spark):
    rows = [ev(f"s{i}", 1, ts(13, i), ingest=ts(10, i, 3)) for i in range(5)]  # client clock +3h
    rows += [ev("late", 1, ts(13, 30), ingest=ts(10, 0, 0, d=2))]  # delivered next day, outvoted
    rows += [ev(f"o{i}", 2, ts(9, i), ingest=ts(9, i, 2)) for i in range(3)]
    rows += [ev("l2", 3, ts(9), ingest=ts(9, d=2))]  # only a late delivery: implausible 24h -> 0
    df = spark.createDataFrame(rows, EV)
    off = T.estimate_clock_offset(df)
    exp = spark.createDataFrame([(1, 3), (2, 0), (3, 0)], "user_id long, clock_offset_h int")
    assert_df_equality(off, exp, ignore_row_order=True, ignore_nullable=True)
    fixed = T.apply_clock_offset(df.filter("user_id = 1 AND event_id = 's0'"), off).select("event_ts")
    assert fixed.collect()[0][0] == ts(10)


def test_sessionise_30_minute_boundary(spark):
    rows = [
        ev("e1", 1, ts(10)),
        ev("e2", 1, ts(10, 30)),  # exactly 30 min later: same session
        ev("e3", 1, ts(11, 0, 1)),  # 30 min 1 s later: new session
        ev("e4", 2, ts(10, 5)),
    ]
    df = spark.createDataFrame(rows, EV).withColumn("event_ts", F.col("client_ts"))
    out = T.sessionise(df).select("event_id", "session_seq")
    exp = spark.createDataFrame([("e1", 1), ("e2", 1), ("e3", 2), ("e4", 1)], "event_id string, session_seq long")
    assert_df_equality(out, exp, ignore_row_order=True, ignore_nullable=True)


def test_session_table_measures(spark):
    rows = [
        ev("e1", 1, ts(10), art=7),
        ev("e2", 1, ts(10, 2), art=7, typ="scroll", depth=40),
        ev("e3", 1, ts(10, 5), art=9, typ="click"),
        ev("e4", 1, ts(10, 6), art=9, typ="ad_impression"),
        ev("e5", 1, ts(12), art=3),
    ]
    df = spark.createDataFrame(rows, EV).withColumn("event_ts", F.col("client_ts"))
    s = T.session_table(T.sessionise(df)).orderBy("session_start").collect()
    assert len(s) == 2
    a = s[0]
    assert (a.n_events, a.n_page_views, a.n_scrolls, a.n_clicks, a.n_ad_impressions) == (4, 1, 1, 1, 1)
    assert (a.landing_article_id, a.exit_article_id, a.n_articles, a.max_scroll_depth) == (7, 9, 2, 40)
    assert a.duration_s == 360.0
    assert s[1].n_events == 1 and s[1].duration_s == 0.0
    # deterministic id
    s2 = T.session_table(T.sessionise(df)).orderBy("session_start").collect()
    assert [r.session_id for r in s] == [r.session_id for r in s2]


def test_bot_flags(spark):
    rows = [ev(f"r{i}", 1, ts(8) + timedelta(seconds=3 * i)) for i in range(200)]  # metronome, browser UA
    rows += [ev("u1", 2, ts(10), ua="NewsScraperBot/2.1")]  # declared bot
    rows += [ev(f"h{i}", 3, ts(10, i * 2)) for i in range(20)]  # human
    df = spark.createDataFrame(rows, EV).withColumn("event_ts", F.col("client_ts"))
    out = T.bot_flags(df).select("user_id", "is_bot", "bot_reason")
    exp = spark.createDataFrame(
        [(1, True, "regular_interval"), (2, True, "user_agent"), (3, False, None)], "user_id long, is_bot boolean, bot_reason string"
    )
    assert_df_equality(out, exp, ignore_row_order=True, ignore_nullable=True)


def test_checksum_is_order_independent(spark):
    a = spark.createDataFrame([(1, "x"), (2, "y")], "k long, v string")
    b = spark.createDataFrame([(2, "y"), (1, "x")], "k long, v string").repartition(3)
    c = spark.createDataFrame([(1, "x"), (2, "z")], "k long, v string")
    assert T.table_checksum(a) == T.table_checksum(b)
    assert T.table_checksum(a) != T.table_checksum(c)
