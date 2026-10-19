"""Pure DataFrame -> DataFrame transformations (unit-tested with chispa)."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from clickstream.config import SESSION_GAP_S

BOT_UA_REGEX = r"(?i)(bot|crawler|spider|headless|scraper|python-requests)"
MAX_PLAUSIBLE_SKEW_H = 14  # anything larger is a late delivery, not a wrong clock
BOT_MIN_EVENTS_PER_DAY = 150
BOT_MAX_GAP_CV = 0.35
BOT_MAX_EVENTS_PER_HOUR = 600.0


def _epoch(c: str) -> F.Column:
    return F.unix_micros(F.col(c)).cast("double") / 1e6


def dedup_deliveries(df: DataFrame) -> DataFrame:
    """Keep the earliest delivery of each event_id (ties broken on the payload, so the result is deterministic)."""
    w = Window.partitionBy("event_id").orderBy(F.col("ingest_ts").asc(), F.col("client_ts").asc())
    return df.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")


def estimate_clock_offset(events: DataFrame) -> DataFrame:
    """Per-user client clock offset in whole hours = the most common round((client - ingest) / 1h).

    Wrong-timezone devices are off by whole hours, network latency is seconds, so rounding removes latency.
    Late deliveries (ingest a day later) are outvoted by on-time events; implausible offsets fall back to 0.
    """
    h = F.round((_epoch("client_ts") - _epoch("ingest_ts")) / 3600.0).cast("int")
    c = events.select("user_id", h.alias("offset_h")).groupBy("user_id", "offset_h").count()
    w = Window.partitionBy("user_id").orderBy(F.col("count").desc(), F.abs(F.col("offset_h")).asc(), F.col("offset_h").asc())
    best = c.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1)
    off = F.when(F.abs(F.col("offset_h")) > MAX_PLAUSIBLE_SKEW_H, F.lit(0)).otherwise(F.col("offset_h"))
    return best.select("user_id", off.alias("clock_offset_h"))


def apply_clock_offset(events: DataFrame, offsets: DataFrame) -> DataFrame:
    """event_ts = client_ts - offset hours (offset 0 when unknown)."""
    df = events.join(offsets, "user_id", "left").fillna({"clock_offset_h": 0})
    shift = F.col("clock_offset_h").cast("long") * F.lit(3_600_000_000)
    return df.withColumn("event_ts", F.timestamp_micros(F.unix_micros("client_ts") - shift))


def bot_flags(events: DataFrame) -> DataFrame:
    """Per-user bot heuristic on corrected events (needs ``event_ts``).

    A user is a bot if its user agent looks automated, or on any day it produced >= 150 events with
    near-constant spacing (gap coefficient of variation < 0.35) or at > 600 events per active hour.
    """
    ua = events.groupBy("user_id").agg(F.max(F.col("user_agent").rlike(BOT_UA_REGEX).cast("int")).alias("ua_bot"))
    d = events.withColumn("_day", F.to_date("event_ts")).withColumn("_t", _epoch("event_ts"))
    w = Window.partitionBy("user_id", "_day").orderBy("_t", "event_id")
    d = d.withColumn("_gap", F.col("_t") - F.lag("_t").over(w))
    day = d.groupBy("user_id", "_day").agg(
        F.count("*").alias("n"),
        F.avg("_gap").alias("gap_mean"),
        F.stddev_pop("_gap").alias("gap_sd"),
        (F.max("_t") - F.min("_t")).alias("span_s"),
    )
    cv = F.col("gap_sd") / F.when(F.col("gap_mean") > 0, F.col("gap_mean"))
    rate = F.col("n") / ((F.col("span_s") + 60.0) / 3600.0)
    day = day.withColumn("regular", (F.col("n") >= BOT_MIN_EVENTS_PER_DAY) & (F.coalesce(cv, F.lit(0.0)) < BOT_MAX_GAP_CV))
    day = day.withColumn("fast", (F.col("n") >= BOT_MIN_EVENTS_PER_DAY) & (rate > BOT_MAX_EVENTS_PER_HOUR))
    beh = day.groupBy("user_id").agg(
        F.max(F.col("regular").cast("int")).alias("regular"),
        F.max(F.col("fast").cast("int")).alias("fast"),
        F.max("n").alias("max_events_per_day"),
    )
    out = ua.join(beh, "user_id")
    reason = (
        F.when(F.col("ua_bot") == 1, "user_agent")
        .when(F.col("regular") == 1, "regular_interval")
        .when(F.col("fast") == 1, "rate")
        .otherwise(F.lit(None).cast("string"))
    )
    return out.select("user_id", reason.isNotNull().alias("is_bot"), reason.alias("bot_reason"), "max_events_per_day")


def sessionise(events: DataFrame, gap_s: int = SESSION_GAP_S) -> DataFrame:
    """Assign ``session_seq`` per user: a new session starts after > gap_s seconds of inactivity."""
    w = Window.partitionBy("user_id").orderBy("event_ts", "event_id")
    t = _epoch("event_ts")
    prev = F.lag(t).over(w)
    new = F.when(prev.isNull() | ((t - prev) > F.lit(float(gap_s))), 1).otherwise(0)
    df = events.withColumn("_new", new)
    df = df.withColumn("session_seq", F.sum("_new").over(w.rowsBetween(Window.unboundedPreceding, Window.currentRow)))
    return df.drop("_new")


SESSION_COLS = [
    "session_id",
    "user_id",
    "session_date",
    "session_start",
    "session_end",
    "duration_s",
    "n_events",
    "n_page_views",
    "n_scrolls",
    "n_clicks",
    "n_ad_impressions",
    "max_scroll_depth",
    "n_articles",
    "landing_article_id",
    "exit_article_id",
    "device",
]


def session_table(sessionised: DataFrame) -> DataFrame:
    """One row per (user, session_seq) with engagement measures and a deterministic session_id."""
    order = F.struct("event_ts", "event_id")
    g = sessionised.groupBy("user_id", "session_seq").agg(
        F.min("event_ts").alias("session_start"),
        F.max("event_ts").alias("session_end"),
        F.count("*").alias("n_events"),
        F.sum((F.col("event_type") == "page_view").cast("int")).alias("n_page_views"),
        F.sum((F.col("event_type") == "scroll").cast("int")).alias("n_scrolls"),
        F.sum((F.col("event_type") == "click").cast("int")).alias("n_clicks"),
        F.sum((F.col("event_type") == "ad_impression").cast("int")).alias("n_ad_impressions"),
        F.max("scroll_depth").alias("max_scroll_depth"),
        F.count_distinct("article_id").alias("n_articles"),
        F.min_by("article_id", order).alias("landing_article_id"),
        F.max_by("article_id", order).alias("exit_article_id"),
        F.min_by("device", order).alias("device"),
    )
    sid = F.substring(F.sha2(F.concat_ws("|", F.col("user_id").cast("string"), F.unix_micros("session_start").cast("string")), 256), 1, 20)
    g = g.withColumn("session_id", sid)
    g = g.withColumn("session_date", F.to_date("session_start"))
    g = g.withColumn("duration_s", F.round(_epoch("session_end") - _epoch("session_start"), 3))
    return g.select(*SESSION_COLS)


def article_daily(events: DataFrame, profiles: DataFrame) -> DataFrame:
    """Human-only engagement per article per (corrected) day."""
    df = apply_clock_offset(events, profiles.select("user_id", "clock_offset_h"))
    df = df.join(profiles.filter(F.col("is_bot")).select("user_id"), "user_id", "left_anti")
    df = df.withColumn("event_date", F.to_date("event_ts"))
    return df.groupBy("event_date", "article_id").agg(
        F.sum((F.col("event_type") == "page_view").cast("int")).alias("page_views"),
        F.sum((F.col("event_type") == "scroll").cast("int")).alias("scrolls"),
        F.sum((F.col("event_type") == "click").cast("int")).alias("clicks"),
        F.sum((F.col("event_type") == "ad_impression").cast("int")).alias("ad_impressions"),
        F.count_distinct("user_id").alias("unique_users"),
    )


def table_checksum(df: DataFrame) -> tuple[int, str]:
    """(row count, order-independent content hash) of a DataFrame."""
    cols = sorted(df.columns)
    r = df.select(F.xxhash64(*[F.col(c) for c in cols]).cast("decimal(38,0)").alias("h")).agg(
        F.count("*").alias("n"), F.sum("h").alias("s"), F.sum(F.col("h") * F.col("h") % 1000000007).alias("s2")
    )
    row = r.collect()[0]
    return int(row["n"]), f"{row['s']}:{row['s2']}"
