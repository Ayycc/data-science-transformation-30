"""End-to-end run: generate -> bronze/silver/gold batches -> idempotency, time travel, evaluation, tuning."""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import pandas as pd
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from clickstream import evaluate, layout, skew
from clickstream import transforms as T
from clickstream.config import BASE_DATE, SESSION_GAP_S, Scale
from clickstream.data import write_landing
from clickstream.lakehouse import Lakehouse


def schema_checks(lh: Lakehouse) -> dict:
    """Schema enforcement (bad type / null key / bad enum rejected) and evolution (app_version added)."""
    s = lh.spark
    out: dict = {}
    good = lh.landing(1).limit(5)
    bad_type = good.withColumn("client_ts", F.date_format("client_ts", "dd/MM/yyyy HH:mm")).withColumn("batch_id", F.lit(99))
    null_key = good.withColumn("event_id", F.lit(None).cast("string")).withColumn("batch_id", F.lit(99))
    bad_enum = good.withColumn("event_type", F.lit("hover")).withColumn("batch_id", F.lit(99))
    v0 = lh.version(lh.bronze)
    for name, df in [("string_timestamp", bad_type), ("null_event_id", null_key), ("unknown_event_type", bad_enum)]:
        try:
            df.write.format("delta").mode("append").save(lh.bronze)
            out[name] = "ACCEPTED"
        except Exception as e:  # noqa: BLE001 - we record the engine's error class
            found = re.findall(r"\[(DELTA_[A-Z_]+)\]", str(e))
            out[name] = "REJECTED: " + (found[0] if found else type(e).__name__)
    out["bronze_version_unchanged"] = lh.version(lh.bronze) == v0
    out["silver_columns"] = [f.name for f in s.read.format("delta").load(lh.silver).schema.fields]
    return out


def time_travel_audit(lh: Lakehouse, versions: dict[int, int], day1: str) -> dict:
    """Compare day-1 sessions as published after batch 1 with the current table (late events revised them)."""
    s = lh.spark
    v1 = s.read.format("delta").option("versionAsOf", versions[1]).load(lh.sessions).filter(F.col("session_date") == day1)
    now = s.read.format("delta").load(lh.sessions).filter(F.col("session_date") == day1)
    a = v1.select("session_id", F.col("row_hash").alias("h1"), F.col("n_events").alias("n1"))
    b = now.select("session_id", F.col("row_hash").alias("h2"), F.col("n_events").alias("n2"))
    j = a.join(b, "session_id", "full_outer")
    r = j.agg(
        F.sum((F.col("h1").isNotNull() & F.col("h2").isNotNull() & (F.col("h1") == F.col("h2"))).cast("long")).alias("unchanged"),
        F.sum((F.col("h1").isNotNull() & F.col("h2").isNotNull() & (F.col("h1") != F.col("h2"))).cast("long")).alias("updated"),
        F.sum((F.col("h1").isNotNull() & F.col("h2").isNull()).cast("long")).alias("removed"),
        F.sum((F.col("h1").isNull() & F.col("h2").isNotNull()).cast("long")).alias("added"),
        F.sum(F.coalesce("n1", F.lit(0))).alias("events_v1"),
        F.sum(F.coalesce("n2", F.lit(0))).alias("events_now"),
    ).collect()[0]
    sql = f"SELECT count(*) FROM delta.`{lh.sessions}` VERSION AS OF {versions[1]} WHERE session_date = DATE'{day1}'"
    return {
        "day": day1,
        "version_after_batch1": versions[1],
        "version_now": lh.version(lh.sessions),
        "sql": sql.replace(lh.sessions, "gold_sessions"),
        "sessions_v1": int(s.sql(sql).collect()[0][0]),
        **{k: int(r[k] or 0) for k in ("unchanged", "updated", "removed", "added", "events_v1", "events_now")},
    }


def history_frame(lh: Lakehouse, path: str, table: str) -> pd.DataFrame:
    h = DeltaTable.forPath(lh.spark, path).history().select("version", "operation", "operationParameters", "operationMetrics")
    rows = []
    for r in h.orderBy("version").collect():
        m = r["operationMetrics"] or {}
        keep = [
            "numOutputRows",
            "numTargetRowsInserted",
            "numTargetRowsUpdated",
            "numTargetRowsDeleted",
            "numTargetRowsCopied",
            "numTargetFilesAdded",
            "numTargetFilesRemoved",
            "numFiles",
            "numAddedFiles",
            "numRemovedFiles",
            "executionTimeMs",
        ]
        rows.append({"table": table, "version": r["version"], "operation": r["operation"], **{k: m.get(k) for k in keep if k in m}})
    return pd.DataFrame(rows)


def run(
    spark: SparkSession, sc: Scale, data_dir: Path, results: Path, skew_repeats: int = 3, layout_repeats: int = 3, fresh: bool = True
) -> dict:
    t_all = time.perf_counter()
    results.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    gen = write_landing(spark, sc, data_dir / "landing")
    timings["generate_s"] = round(time.perf_counter() - t0, 1)

    delta_root = data_dir / "delta"
    if fresh and delta_root.exists():
        shutil.rmtree(delta_root)
    lh = Lakehouse(spark, delta_root, str(data_dir / "landing" / "deliveries"), sc.days)
    lh.create_tables()

    batches = {}
    versions: dict[int, int] = {}
    checks: dict[int, dict] = {}
    for b in range(1, sc.days + 1):
        info = lh.run_batch(b)
        info["silver_merge"] = lh.last_operation_metrics(lh.silver)
        info["gold_sessions_merge"] = lh.last_operation_metrics(lh.sessions)
        versions[b] = lh.version(lh.sessions)
        batches[b] = info
        print(f"batch {b}: {json.dumps({k: v for k, v in info.items() if not isinstance(v, dict)})}", flush=True)
    checks[sc.days] = lh.gold_checksums()
    timings["batches_s"] = round(sum(batches[b]["bronze_s"] + batches[b]["silver_s"] + batches[b]["gold_s"] for b in batches), 1)

    # --- idempotency: replay batch 2 (bronze txn dedup, silver no-op MERGE, gold no-op MERGE)
    t0 = time.perf_counter()
    replay = {}
    for b in (2,):
        v_before = {n: lh.version(p) for n, p in [("bronze", lh.bronze), ("silver", lh.silver), ("gold_sessions", lh.sessions)]}
        info = lh.run_batch(b)
        rec = {"seconds": round(info["bronze_s"] + info["silver_s"] + info["gold_s"], 1)}
        for n, p in [("bronze", lh.bronze), ("silver", lh.silver), ("gold_sessions", lh.sessions)]:
            v_after = lh.version(p)
            rows_changed = 0
            if v_after != v_before[n]:
                m = lh.last_operation_metrics(p)
                rows_changed = sum(
                    m.get(k, 0)
                    for k in ("numTargetRowsInserted", "numTargetRowsUpdated", "numTargetRowsDeleted", "numOutputRows")
                    if n == "bronze" or k != "numOutputRows"
                )
            rec[n] = {"version_before": v_before[n], "version_after": v_after, "rows_changed": rows_changed}
        replay[b] = rec
    after = lh.gold_checksums()
    idem = {
        "replayed_batches": list(replay),
        "gold_identical": after == checks[sc.days],
        "checksums": {k: {"rows": v[0], "hash": v[1]} for k, v in after.items()},
        "replay": replay,
    }
    timings["idempotency_s"] = round(time.perf_counter() - t0, 1)

    # --- incremental == full recompute from silver
    t0 = time.perf_counter()
    all_users = spark.read.format("delta").load(lh.silver).select("user_id").distinct()
    prof_full, sess_full = lh.recompute_users(all_users)
    full_ck = T.table_checksum(sess_full)
    inc_ck = after["gold_sessions"]
    prof_ck = T.table_checksum(prof_full)
    lh.release()
    equivalence = {"sessions_equal": full_ck == inc_ck, "profiles_equal": prof_ck == after["gold_user_profiles"], "full_rows": full_ck[0]}
    timings["full_recompute_check_s"] = round(time.perf_counter() - t0, 1)

    # --- schema enforcement / evolution
    schema = schema_checks(lh)

    # --- time travel audit
    day1 = BASE_DATE
    tt = time_travel_audit(lh, versions, day1)

    # --- evaluation vs ground truth
    t0 = time.perf_counter()
    truth_users = spark.read.parquet(str(data_dir / "landing" / "truth_users"))
    truth_sessions = spark.read.parquet(str(data_dir / "landing" / "truth_sessions"))
    deliveries = spark.read.parquet(str(data_dir / "landing" / "deliveries"))
    sessions = spark.read.format("delta").load(lh.sessions)
    profiles = spark.read.format("delta").load(lh.profiles)
    silver = spark.read.format("delta").load(lh.silver)
    bots = evaluate.bot_detection(profiles, truth_users)
    rec = evaluate.session_recovery(spark, sessions.filter(~F.col("is_bot")), truth_sessions, truth_users)
    naive = evaluate.naive_session_recovery(deliveries, truth_sessions, truth_users)
    counts = {
        "bronze_rows": spark.read.format("delta").load(lh.bronze).count(),
        "silver_rows": silver.count(),
        "gold_sessions": sessions.count(),
        "gold_human_sessions": sessions.filter(~F.col("is_bot")).count(),
        "gold_bot_sessions": sessions.filter(F.col("is_bot")).count(),
        "gold_article_daily_rows": spark.read.format("delta").load(lh.article_daily).count(),
        "users_with_clock_offset": profiles.filter(F.col("clock_offset_h") != 0).count(),
        "true_skewed_users_active": truth_users.filter(F.col("skew_hours") != 0).join(profiles, "user_id", "left_semi").count(),
        "clock_offset_correct": profiles.join(truth_users, "user_id").filter(F.col("clock_offset_h") == F.col("skew_hours")).count(),
        "skewed_users_offset_correct": profiles.join(truth_users, "user_id")
        .filter((F.col("skew_hours") != 0) & (F.col("clock_offset_h") == F.col("skew_hours")))
        .count(),
        "profiles": profiles.count(),
    }
    counts["duplicates_removed"] = counts["bronze_rows"] - counts["silver_rows"]
    human = sessions.filter(~F.col("is_bot"))
    sess_stats = human.agg(
        F.avg("duration_s").alias("mean_duration_s"),
        F.percentile_approx("duration_s", 0.5, 10000).alias("median_duration_s"),
        F.avg("n_events").alias("mean_events"),
        F.avg((F.col("n_events") == 1).cast("double")).alias("bounce_rate"),
    ).collect()[0]
    bot_share = silver.join(profiles.filter("is_bot").select("user_id"), "user_id", "left_semi").count() / counts["silver_rows"]
    timings["evaluate_s"] = round(time.perf_counter() - t0, 1)

    # --- skew benchmark on all page views (cached, so every variant pays the same, tiny scan cost)
    t0 = time.perf_counter()
    dim = spark.read.parquet(str(data_dir / "landing" / "dim_campaign_targets")).persist()
    views = silver.filter(F.col("event_type") == "page_view").select("event_id", "article_id").persist()
    n_views = views.count()
    dim.count()
    viral_share = views.filter(F.col("article_id") < len(sc.viral_weights)).count() / max(n_views, 1)
    top1_share = views.filter(F.col("article_id") == 0).count() / max(n_views, 1)
    sk = skew.benchmark(spark, views, dim, repeats=skew_repeats)
    sk["viral_share_of_views"] = round(viral_share, 4)
    sk["top_article_share_of_views"] = round(top1_share, 4)
    sk["dim_rows"] = dim.count()
    views.unpersist()
    dim.unpersist()
    for v, plan in sk.pop("plans").items():
        (results / f"explain_skew_{v}.txt").write_text(plan)
    timings["skew_benchmark_s"] = round(time.perf_counter() - t0, 1)

    # --- layout benchmark (OPTIMIZE + ZORDER) on silver
    t0 = time.perf_counter()
    probe_user = int(
        human.groupBy("user_id").agg(F.sum("n_events").alias("n")).orderBy(F.desc("n"), "user_id").limit(1).collect()[0]["user_id"]
    )
    probe_article = 1000
    lay = layout.benchmark(spark, lh.silver, probe_user, probe_article, day=day1, repeats=layout_repeats)
    timings["layout_benchmark_s"] = round(time.perf_counter() - t0, 1)

    # --- exports
    hist = pd.concat(
        [history_frame(lh, p, n) for n, p in [("bronze_events", lh.bronze), ("silver_events", lh.silver), ("gold_sessions", lh.sessions)]]
    )
    hist.to_csv(results / "delta_history.csv", index=False)
    human.select("duration_s", "n_events").sample(
        fraction=min(1.0, 200_000 / max(counts["gold_human_sessions"], 1)), seed=sc.seed
    ).toPandas().to_csv(results / "_session_sample.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    top_articles = (
        spark.read.format("delta")
        .load(lh.article_daily)
        .groupBy("article_id")
        .agg(F.sum("page_views").alias("page_views"), F.sum("unique_users").alias("user_days"))
        .orderBy(F.desc("page_views"), "article_id")
        .limit(10)
        .toPandas()
    )
    top_articles.to_csv(results / "top_articles.csv", index=False)

    timings["total_s"] = round(time.perf_counter() - t_all, 1)
    metrics = {
        "data_source": "SYNTHETIC (seeded Spark generator, src/clickstream/generate.py); no public download attempted",
        "scale": sc.name,
        "seed": sc.seed,
        "session_gap_minutes": SESSION_GAP_S // 60,
        "generator": gen,
        "batches": batches,
        "gold_versions_after_batch": versions,
        "idempotency": idem,
        "incremental_equals_full_recompute": equivalence,
        "schema": schema,
        "time_travel_audit": tt,
        "bot_detection": bots,
        "bot_share_of_silver_events": round(bot_share, 4),
        "session_recovery": rec,
        "naive_session_recovery": naive,
        "counts": counts,
        "human_sessions": {k: round(float(sess_stats[k]), 4) for k in sess_stats.asDict()},
        "skew_benchmark": sk,
        "layout_benchmark": lay,
        "timings": timings,
        "spark": {"version": spark.version, "cores": spark.sparkContext.defaultParallelism},
    }
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics
