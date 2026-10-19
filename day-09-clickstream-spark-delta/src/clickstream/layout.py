"""File-layout benchmark: selective reads on silver before and after OPTIMIZE ... ZORDER BY."""

from __future__ import annotations

import statistics
import time

from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from clickstream.uimetrics import SparkUI


def _file_stats(spark: SparkSession, path: str) -> dict:
    d = spark.sql(f"DESCRIBE DETAIL delta.`{path}`").collect()[0]
    return {"num_files": int(d["numFiles"]), "size_mb": round(int(d["sizeInBytes"]) / 2**20, 1)}


def _probe(spark: SparkSession, path: str, label: str, predicate: str, repeats: int) -> dict:
    sc = spark.sparkContext
    ui = SparkUI(spark)
    ts = []
    rows = None
    for i in range(repeats):
        desc = f"layout-{label}-{i}"
        sc.setJobDescription(desc)
        t0 = time.perf_counter()
        rows = spark.read.format("delta").load(path).filter(predicate).agg(F.count("*").alias("n")).collect()[0]["n"]
        ts.append(time.perf_counter() - t0)
        sc.setJobDescription(None)
    m = ui.sql_scan_metrics(f"layout-{label}-0") if ui.available else {}
    return {
        "predicate": predicate,
        "rows": int(rows),
        "runtime_s_median": round(statistics.median(ts), 3),
        "files_read": int(m.get("number of files read", -1)),
        "mb_read": round(m.get("size of files read", -1.0), 1),
    }


def benchmark(
    spark: SparkSession,
    path: str,
    user_id: int,
    article_id: int,
    day: str,
    repeats: int = 3,
    max_file_mb: int = 16,
    zorder: tuple[str, ...] = ("user_id",),
) -> dict:
    """Probe user / article / one-day reads, OPTIMIZE ... ZORDER BY ``zorder`` into ``max_file_mb`` files, probe again."""
    probes = {"user_lookup": f"user_id = {user_id}", "article_lookup": f"article_id = {article_id}", "one_day": f"event_date = DATE'{day}'"}
    before = {"files": _file_stats(spark, path)}
    for k, pred in probes.items():
        before[k] = _probe(spark, path, f"before-{k}", pred, repeats)
    prev = spark.conf.get("spark.databricks.delta.optimize.maxFileSize", None)
    spark.conf.set("spark.databricks.delta.optimize.maxFileSize", str(max_file_mb * 2**20))
    t0 = time.perf_counter()
    m = DeltaTable.forPath(spark, path).optimize().executeZOrderBy(*zorder).collect()[0]
    opt_s = time.perf_counter() - t0
    if prev is None:
        spark.conf.unset("spark.databricks.delta.optimize.maxFileSize")
    else:
        spark.conf.set("spark.databricks.delta.optimize.maxFileSize", prev)
    after = {"files": _file_stats(spark, path)}
    for k, pred in probes.items():
        after[k] = _probe(spark, path, f"after-{k}", pred, repeats)
    metrics = m["metrics"].asDict() if m and "metrics" in m else {}
    return {
        "before": before,
        "after": after,
        "optimize_s": round(opt_s, 1),
        "optimize_max_file_mb": max_file_mb,
        "zorder_by": list(zorder),
        "optimize_files_added": int(metrics.get("numFilesAdded", -1)),
        "optimize_files_removed": int(metrics.get("numFilesRemoved", -1)),
    }
