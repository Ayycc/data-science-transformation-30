"""Skewed-join benchmark: page views x ad-campaign targeting, with and without skew mitigation.

The five viral articles carry ~30% of human page views *and* were sold to 24 campaigns each, so after
the join their shuffle partitions hold most of the work. Broadcast joins are disabled to stand in for a
dimension too large to broadcast; every variant is a sort-merge join on ``article_id``.

Variants
* ``baseline``  AQE on (coalescing only), skew-join handling off
* ``aqe_skew``  AQE skew-join: oversized partitions are split and the matching dimension side replicated
* ``salted``    manual salting of the hot keys found at run time (top-k article_ids), dimension replicated
"""

from __future__ import annotations

import statistics
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from clickstream.uimetrics import SparkUI

COMMON_CONF = {
    "spark.sql.autoBroadcastJoinThreshold": "-1",
    "spark.sql.adaptive.autoBroadcastJoinThreshold": "-1",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "false",
    "spark.sql.adaptive.advisoryPartitionSizeInBytes": "512KB",
    "spark.sql.adaptive.skewJoin.skewedPartitionFactor": "3",
    "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "1MB",
}
VARIANT_CONF = {
    "baseline": {"spark.sql.adaptive.skewJoin.enabled": "false"},
    "aqe_skew": {"spark.sql.adaptive.skewJoin.enabled": "true"},
    "salted": {"spark.sql.adaptive.skewJoin.enabled": "false"},
}


def _query(views: DataFrame, dim: DataFrame) -> DataFrame:
    j = views.join(dim, "article_id")
    return j.groupBy("campaign_id").agg(F.count("*").alias("attributed_views"), F.sum("cpm_cents").alias("revenue_milli_cents"))


def salted_query(views: DataFrame, dim: DataFrame, hot_keys: list[int], n_salts: int) -> DataFrame:
    hot = F.col("article_id").isin(hot_keys)
    v = views.withColumn("salt", F.when(hot, F.pmod(F.xxhash64("event_id"), F.lit(n_salts))).otherwise(F.lit(0)).cast("int"))
    d = dim.withColumn("salt", F.when(hot, F.sequence(F.lit(0), F.lit(n_salts - 1))).otherwise(F.array(F.lit(0))))
    d = d.withColumn("salt", F.explode("salt"))
    j = v.join(d, ["article_id", "salt"])
    return j.groupBy("campaign_id").agg(F.count("*").alias("attributed_views"), F.sum("cpm_cents").alias("revenue_milli_cents"))


def run_variant(spark: SparkSession, views: DataFrame, dim: DataFrame, variant: str, n_salts: int, top_k: int) -> tuple[DataFrame, list]:
    for k, v in {**COMMON_CONF, **VARIANT_CONF[variant]}.items():
        spark.conf.set(k, v)
    if variant == "salted":
        hot = [int(r[0]) for r in views.groupBy("article_id").count().orderBy(F.desc("count"), "article_id").limit(top_k).collect()]
        q = salted_query(views, dim, hot, n_salts)
    else:
        q = _query(views, dim)
    q = q.orderBy("campaign_id")
    rows = q.collect()
    return q, rows


def benchmark(spark: SparkSession, views: DataFrame, dim: DataFrame, repeats: int = 3, n_salts: int = 24, top_k: int = 5) -> dict:
    sc = spark.sparkContext
    ui = SparkUI(spark)
    saved = {k: spark.conf.get(k, None) for k in {**COMMON_CONF, **VARIANT_CONF["baseline"]}}
    out: dict = {"variants": {}, "plans": {}}
    reference = None
    times: dict[str, list[float]] = {v: [] for v in VARIANT_CONF}
    stage_stats: dict[str, list] = {}
    # warm-up so the first variant does not pay JIT / file-cache costs
    _query(views, dim).count()
    for rep in range(repeats):
        for variant in VARIANT_CONF:
            group = f"skew-{variant}-{rep}"
            sc.setJobGroup(group, group)
            t0 = time.perf_counter()
            q, rows = run_variant(spark, views, dim, variant, n_salts, top_k)
            times[variant].append(time.perf_counter() - t0)
            sc.setJobGroup("", "")
            res = [(r["campaign_id"], r["attributed_views"], r["revenue_milli_cents"]) for r in rows]
            if reference is None:
                reference = res
            out.setdefault("results_identical", True)
            out["results_identical"] = out["results_identical"] and res == reference
            if rep == 0:
                out["plans"][variant] = q._jdf.queryExecution().executedPlan().toString()
                if ui.available:
                    stage_stats[variant] = ui.stages_for_group(group)
    for variant, ts in times.items():
        st = stage_stats.get(variant, [])
        join_stage = max(st, key=lambda s: s["shuffle_read_mb"]) if st else {}
        out["variants"][variant] = {
            "runtime_s_runs": [round(t, 2) for t in ts],
            "runtime_s_median": round(statistics.median(ts), 2),
            "slowest_stage": join_stage,
            "stages": st,
            "plan_has_skew_split": "skew=true" in out["plans"].get(variant, ""),
        }
    base = out["variants"]["baseline"]["runtime_s_median"]
    for variant in VARIANT_CONF:
        out["variants"][variant]["speedup_vs_baseline"] = round(base / out["variants"][variant]["runtime_s_median"], 2)
    out["rows_joined_input"] = views.count()
    out["n_salts"] = n_salts
    out["top_k_hot_keys"] = top_k
    out["conf"] = COMMON_CONF
    for k, v in saved.items():
        if v is None:
            spark.conf.unset(k)
        else:
            spark.conf.set(k, v)
    return out
