"""Seeded SYNTHETIC clickstream generator, written in Spark so it scales to ~50M events.

All randomness comes from ``xxhash64(seed, stream, keys...)``, so the output is identical for a given
seed regardless of partitioning or core count.

What it plants (and what the pipeline must cope with):

* humans: sessions separated by >= 31 minutes of inactivity, events inside a session < 30 minutes apart
  (so the true sessions are known exactly);
* bots: one long crawl a day at a near-constant interval; ~60% announce themselves in the user agent,
  the rest pretend to be a browser;
* clock skew: ~3% of devices report client time shifted by a whole number of hours (wrong timezone);
* late arrivals: ~1.5% of events are delivered in the next day's batch;
* duplicate deliveries: ~2% of events are delivered twice, up to 6 hours apart (sometimes across batches);
* heavy hitters: ~30% of human events land on 5 viral articles (the top one alone gets ~18%).
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from clickstream.config import BASE_DATE, Scale

TWO53 = float(1 << 53)
SLOT_S = 90 * 60  # one human session per 90-minute slot -> >= 31 min gap between sessions
SLOTS_PER_DAY = 16
MAX_SESSION_S = 29 * 60
SKEW_HOURS = [-8, -5, -3, 1, 2, 9]
DEVICES = ["mobile", "desktop", "tablet"]
HUMAN_UA = {
    "mobile": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) Mobile/15E148",
    "desktop": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130.0 Safari/537.36",
    "tablet": "Mozilla/5.0 (Linux; Android 15; Tab) Chrome/130.0 Safari/537.36",
}
BOT_UA = ["Mozilla/5.0 (compatible; NewsScraperBot/2.1)", "python-requests/2.32 crawler", "HeadlessChrome/130.0 spider"]
EVENT_TYPES = ["page_view", "scroll", "click", "ad_impression"]


def u(seed: int, stream: int, *cols: Column | str) -> Column:
    """Deterministic uniform [0, 1) from a hash of (seed, stream, cols)."""
    return F.pmod(F.xxhash64(F.lit(seed), F.lit(stream), *cols), F.lit(1 << 53)).cast("double") / F.lit(TWO53)


def _base_epoch() -> Column:
    return F.unix_timestamp(F.lit(f"{BASE_DATE} 00:00:00")).cast("double")


def users(spark: SparkSession, sc: Scale) -> DataFrame:
    """One row per user: bot flag, bot flavour, device, clock skew (hours). Ground truth."""
    s = sc.seed
    df = spark.range(sc.n_users).withColumnRenamed("id", "user_id")
    df = df.withColumn("is_bot", u(s, 1, "user_id") < F.lit(sc.bot_frac))
    dev_idx = (u(s, 2, "user_id") * 3).cast("int")
    df = df.withColumn("device", F.element_at(F.array(*[F.lit(d) for d in DEVICES]), dev_idx + 1))
    skewed = (u(s, 3, "user_id") < F.lit(sc.skew_user_frac)) & ~F.col("is_bot")
    sk_idx = (u(s, 4, "user_id") * len(SKEW_HOURS)).cast("int")
    df = df.withColumn("skew_hours", F.when(skewed, F.element_at(F.array(*[F.lit(h) for h in SKEW_HOURS]), sk_idx + 1)).otherwise(0))
    # bot flavour: 0 = declared UA, 1 = disguised + metronome interval, 2 = disguised + random interval
    r = u(s, 5, "user_id")
    df = df.withColumn("bot_kind", F.when(~F.col("is_bot"), F.lit(None)).when(r < 0.6, 0).when(r < 0.85, 1).otherwise(2))
    ua_human = F.create_map(*[x for kv in HUMAN_UA.items() for x in (F.lit(kv[0]), F.lit(kv[1]))])[F.col("device")]
    ua_bot = F.element_at(F.array(*[F.lit(x) for x in BOT_UA]), (u(s, 6, "user_id") * len(BOT_UA)).cast("int") + 1)
    df = df.withColumn("user_agent", F.when(F.col("bot_kind") == 0, ua_bot).otherwise(ua_human))
    return df


def true_sessions(spark: SparkSession, sc: Scale) -> DataFrame:
    """Ground-truth human sessions: (user_id, day, slot) -> start epoch seconds, duration, n_events."""
    s = sc.seed
    us = users(spark, sc).filter(~F.col("is_bot")).select("user_id")
    days = spark.range(sc.days).withColumnRenamed("id", "day")
    ud = us.crossJoin(F.broadcast(days))
    n_s = F.least(F.floor(-F.log(F.lit(1.0) - u(s, 10, "user_id", "day")) * F.lit(sc.mean_sessions_per_day)), F.lit(SLOTS_PER_DAY))
    ud = ud.withColumn("n_sessions", n_s.cast("int")).filter(F.col("n_sessions") > 0)
    start_slot = F.floor(u(s, 11, "user_id", "day") * (F.lit(SLOTS_PER_DAY) - F.col("n_sessions") + 1)).cast("int")
    ud = ud.withColumn("start_slot", start_slot)
    se = ud.withColumn("k", F.explode(F.sequence(F.lit(0), F.col("n_sessions") - 1)))
    se = se.withColumn("slot", F.col("start_slot") + F.col("k"))
    start = _base_epoch() + F.col("day") * 86400.0 + F.col("slot") * float(SLOT_S) + u(s, 12, "user_id", "day", "slot") * 1800.0
    dur = F.least(-F.log(F.lit(1.0) - u(s, 13, "user_id", "day", "slot")) * 480.0, F.lit(float(MAX_SESSION_S)))
    n_ev = F.least(
        F.lit(1) + F.floor(-F.log(F.lit(1.0) - u(s, 14, "user_id", "day", "slot")) * F.lit(sc.mean_events_per_session)), F.lit(80)
    )
    se = se.select("user_id", "day", "slot", F.round(start, 3).alias("start_s"), dur.alias("dur_s"), n_ev.cast("int").alias("n_events"))
    return se


def _article_for(s: int, sc: Scale, *keys: str) -> Column:
    viral_cut = []
    acc = 0.0
    for w in sc.viral_weights:
        acc += w
        viral_cut.append(acc)
    rv = u(s, 30, *keys)
    pick = u(s, 31, *keys)
    viral = F.lit(len(viral_cut) - 1)
    for i in reversed(range(len(viral_cut))):
        viral = F.when(pick < F.lit(viral_cut[i]), F.lit(i)).otherwise(viral)
    # long tail: log-uniform over the remaining catalogue (power-law-ish popularity)
    tail = F.lit(len(sc.viral_weights)) + F.floor(F.pow(F.lit(float(sc.n_articles - len(sc.viral_weights))), u(s, 32, *keys))) - 1
    return F.when(rv < F.lit(sc.viral_share), viral).otherwise(tail).cast("long")


def human_events(spark: SparkSession, sc: Scale) -> DataFrame:
    s = sc.seed
    se = true_sessions(spark, sc)
    ev = se.withColumn("idx", F.explode(F.sequence(F.lit(0), F.col("n_events") - 1)))
    offset = F.when(F.col("idx") == 0, F.lit(0.0)).otherwise(u(s, 20, "user_id", "day", "slot", "idx") * F.col("dur_s"))
    ev = ev.withColumn("true_s", F.col("start_s") + offset)
    r = u(s, 21, "user_id", "day", "slot", "idx")
    etype = (
        F.when(F.col("idx") == 0, "page_view")
        .when(r < 0.50, "page_view")
        .when(r < 0.72, "scroll")
        .when(r < 0.88, "click")
        .otherwise("ad_impression")
    )
    ev = ev.withColumn("event_type", etype)
    ev = ev.withColumn("article_id", _article_for(s, sc, "user_id", "day", "slot", "idx"))
    ev = ev.withColumn("event_key", F.concat_ws("-", F.lit("h"), "user_id", "day", "slot", "idx"))
    ev = ev.withColumn("true_session_key", F.concat_ws("-", "user_id", "day", "slot"))
    return ev.select("user_id", "day", "true_s", "event_type", "article_id", "event_key", "true_session_key")


def bot_events(spark: SparkSession, sc: Scale) -> DataFrame:
    s = sc.seed
    bots = users(spark, sc).filter(F.col("is_bot")).select("user_id", "bot_kind")
    days = spark.range(sc.days).withColumnRenamed("id", "day")
    bd = bots.crossJoin(F.broadcast(days))
    lo, hi = sc.bot_events_per_day
    bd = bd.withColumn("n", (F.lit(lo) + F.floor(u(s, 40, "user_id", "day") * (hi - lo))).cast("int"))
    bd = bd.withColumn("interval", F.lit(2.0) + u(s, 41, "user_id") * 3.0)
    bd = bd.withColumn("start_s", _base_epoch() + F.col("day") * 86400.0 + u(s, 42, "user_id", "day") * 10 * 3600.0)
    ev = bd.withColumn("idx", F.explode(F.sequence(F.lit(0), F.col("n") - 1)))
    jitter = (u(s, 43, "user_id", "day", "idx") - 0.5) * 0.2 * F.col("interval")
    rand_gap_pos = -F.log(F.lit(1.0) - u(s, 44, "user_id", "day", "idx")) * F.col("interval")  # used for kind 2
    regular = F.col("start_s") + F.col("idx") * F.col("interval") + jitter
    # kind 2: approximate a Poisson process by perturbing each slot with an exponential draw
    poisson = F.col("start_s") + F.col("idx") * F.col("interval") + rand_gap_pos
    ev = ev.withColumn("true_s", F.when(F.col("bot_kind") == 2, poisson).otherwise(regular))
    ev = ev.withColumn("event_type", F.when(u(s, 45, "user_id", "day", "idx") < 0.9, "page_view").otherwise("scroll"))
    ev = ev.withColumn("article_id", F.floor(u(s, 46, "user_id", "day", "idx") * sc.n_articles).cast("long"))
    ev = ev.withColumn("event_key", F.concat_ws("-", F.lit("b"), "user_id", "day", "idx"))
    ev = ev.withColumn("true_session_key", F.concat_ws("-", F.lit("bot"), "user_id", "day"))
    return ev.select("user_id", "day", "true_s", "event_type", "article_id", "event_key", "true_session_key")


def raw_deliveries(spark: SparkSession, sc: Scale) -> DataFrame:
    """All deliveries (incl. duplicates) with bronze columns, ground-truth columns prefixed ``truth_``."""
    s = sc.seed
    ev = human_events(spark, sc).unionByName(bot_events(spark, sc))
    us = users(spark, sc).select("user_id", "is_bot", "device", "skew_hours", "user_agent")
    ev = ev.join(us, "user_id")
    ev = ev.withColumn("event_id", F.substring(F.sha2(F.concat_ws("|", F.lit(s), "event_key"), 256), 1, 24))
    latency = u(s, 50, "event_key") * 5.0
    late = (u(s, 51, "event_key") < F.lit(sc.late_frac)) & (F.col("day") < F.lit(sc.days - 1))
    # late events arrive between 00:30 and 12:00 of the next day
    next_day = _base_epoch() + (F.col("day") + 1) * 86400.0 + 1800.0 + u(s, 52, "event_key") * 41400.0
    ev = ev.withColumn("ingest_s", F.when(late, F.greatest(next_day, F.col("true_s") + latency)).otherwise(F.col("true_s") + latency))
    ev = ev.withColumn("client_s", F.col("true_s") + F.col("skew_hours") * 3600.0)
    scroll = F.when(F.col("event_type") == "scroll", F.floor(u(s, 53, "event_key") * 100).cast("int") + 1)
    ev = ev.withColumn("scroll_depth", scroll)
    ev = ev.withColumn("delivery", F.lit(0))
    dup = ev.filter(u(s, 54, "event_key") < F.lit(sc.dup_frac))
    dup = dup.withColumn("ingest_s", F.col("ingest_s") + 1.0 + u(s, 55, "event_key") * 6 * 3600.0).withColumn("delivery", F.lit(1))
    allv = ev.unionByName(dup)
    # batches are ingest days; anything ingested after the last day is folded into the last batch
    batch = F.least(F.floor((F.col("ingest_s") - _base_epoch()) / 86400.0).cast("int") + 1, F.lit(sc.days))
    allv = allv.withColumn("batch_id", batch)
    # schema evolution: from the last batch on, the app starts sending app_version
    return allv.select(
        "event_id",
        "user_id",
        "article_id",
        "event_type",
        F.timestamp_seconds(F.round("client_s", 3)).alias("client_ts"),
        F.timestamp_seconds(F.round("ingest_s", 3)).alias("ingest_ts"),
        "device",
        "user_agent",
        "scroll_depth",
        "batch_id",
        F.col("is_bot").alias("truth_is_bot"),
        F.timestamp_seconds(F.round("true_s", 3)).alias("truth_ts"),
        F.col("true_session_key").alias("truth_session_key"),
        F.col("delivery").alias("truth_delivery"),
    )


LANDING_COLS = ["event_id", "user_id", "article_id", "event_type", "client_ts", "ingest_ts", "device", "user_agent", "scroll_depth"]


def landing_batch(raw: DataFrame, batch_id: int, last_batch: int) -> DataFrame:
    """The payload as the collector writes it for one batch (no truth columns)."""
    df = raw.filter(F.col("batch_id") == batch_id).select(*LANDING_COLS)
    if batch_id == last_batch:
        df = df.withColumn("app_version", F.concat(F.lit("7."), (F.pmod(F.col("user_id"), F.lit(5)) + 1).cast("string")))
    return df


def campaign_targets(spark: SparkSession, sc: Scale) -> DataFrame:
    """Ad-campaign targeting dimension: viral articles were sold to many campaigns, the rest to 1-3."""
    s = sc.seed
    a = spark.range(sc.n_articles).withColumnRenamed("id", "article_id")
    n = F.when(F.col("article_id") < len(sc.viral_weights), F.lit(sc.campaigns_per_viral_article)).otherwise(
        F.lit(1) + F.floor(u(s, 60, "article_id") * 3).cast("int")
    )
    a = a.withColumn("k", F.explode(F.sequence(F.lit(1), n)))
    a = a.withColumn("campaign_id", F.floor(u(s, 61, "article_id", "k") * 500).cast("int"))
    a = a.withColumn("cpm_cents", (F.lit(50) + F.floor(u(s, 62, "article_id", "k") * 950)).cast("long"))
    return a.select("article_id", "campaign_id", "cpm_cents")
