"""Score the pipeline against the generator's ground truth (possible only because the data is SYNTHETIC)."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from clickstream import transforms as T


def bot_detection(profiles: DataFrame, truth_users: DataFrame) -> dict:
    j = profiles.select("user_id", F.col("is_bot").alias("pred"), "bot_reason").join(
        truth_users.select("user_id", F.col("is_bot").alias("true"), "bot_kind"), "user_id"
    )
    c = j.agg(
        F.sum((F.col("pred") & F.col("true")).cast("long")).alias("tp"),
        F.sum((F.col("pred") & ~F.col("true")).cast("long")).alias("fp"),
        F.sum((~F.col("pred") & F.col("true")).cast("long")).alias("fn"),
        F.sum((~F.col("pred") & ~F.col("true")).cast("long")).alias("tn"),
    ).collect()[0]
    tp, fp, fn, tn = (int(c[k] or 0) for k in ("tp", "fp", "fn", "tn"))
    by_kind = {
        str(r["bot_kind"]): {"bots": int(r["n"]), "caught": int(r["caught"])}
        for r in j.filter("true")
        .groupBy("bot_kind")
        .agg(F.count("*").alias("n"), F.sum(F.col("pred").cast("int")).alias("caught"))
        .collect()
    }
    reasons = {str(r["bot_reason"]): int(r["count"]) for r in j.filter("pred").groupBy("bot_reason").count().collect()}
    return {
        "users_scored": tp + fp + fn + tn,
        "true_bots": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "recall_by_bot_kind": by_kind,
        "flag_reasons": reasons,
    }


def _match_sessions(found: DataFrame, truth: DataFrame) -> dict:
    """A true session is recovered if a found session of that user starts within 1 s and has the same event count."""
    t = truth.select(
        "user_id",
        F.col("start_s").alias("t_start"),
        F.col("n_events").alias("t_n"),
    )
    f = found.select("user_id", (F.unix_micros("session_start") / 1e6).alias("f_start"), F.col("n_events").alias("f_n"))
    j = t.join(f, "user_id").filter(F.abs(F.col("f_start") - F.col("t_start")) < 1.0)
    exact = j.filter(F.col("f_n") == F.col("t_n")).count()
    start_only = j.select("user_id", "t_start").distinct().count()
    n_true = t.count()
    n_found = f.count()
    return {
        "true_sessions": n_true,
        "found_sessions": n_found,
        "start_matched": start_only,
        "exact_matched": exact,
        "exact_recovery_rate": round(exact / n_true, 6) if n_true else None,
        "found_over_true": round(n_found / n_true, 4) if n_true else None,
    }


def session_recovery(spark: SparkSession, sessions: DataFrame, truth_sessions: DataFrame, truth_users: DataFrame) -> dict:
    humans = truth_users.filter(~F.col("is_bot")).select("user_id")
    found = sessions.join(humans, "user_id", "left_semi")
    return _match_sessions(found, truth_sessions)


def naive_session_recovery(deliveries: DataFrame, truth_sessions: DataFrame, truth_users: DataFrame) -> dict:
    """Same sessionisation on the raw deliveries: no dedup, no clock correction (client time as-is)."""
    humans = truth_users.filter(~F.col("is_bot")).select("user_id")
    raw = deliveries.join(humans, "user_id", "left_semi").withColumn("event_ts", F.col("client_ts"))
    raw = raw.select("event_id", "user_id", "article_id", "event_type", "event_ts", "device", "scroll_depth")
    found = T.session_table(T.sessionise(raw))
    return _match_sessions(found, truth_sessions)
