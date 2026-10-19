"""Write the SYNTHETIC landing zone (all deliveries incl. ground truth) and the committed sample."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from clickstream.config import Scale
from clickstream.generate import campaign_targets, raw_deliveries, true_sessions, users


def write_landing(spark: SparkSession, sc: Scale, out: Path, force: bool = False) -> dict:
    """Generate deliveries to ``out/deliveries`` (parquet by batch_id) plus truth tables. Skips if already present."""
    marker = out / "_GENERATED.json"
    if marker.exists() and not force:
        info = json.loads(marker.read_text())
        if info.get("config") == repr(sc):
            return info
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    t0 = time.perf_counter()
    raw_deliveries(spark, sc).write.mode("overwrite").partitionBy("batch_id").parquet(str(out / "deliveries"))
    users(spark, sc).write.mode("overwrite").parquet(str(out / "truth_users"))
    true_sessions(spark, sc).write.mode("overwrite").parquet(str(out / "truth_sessions"))
    campaign_targets(spark, sc).write.mode("overwrite").parquet(str(out / "dim_campaign_targets"))
    d = spark.read.parquet(str(out / "deliveries"))
    agg = d.agg(
        F.count("*").alias("deliveries"),
        F.sum((F.col("truth_delivery") == 1).cast("long")).alias("duplicate_deliveries"),
        F.sum(F.col("truth_is_bot").cast("long")).alias("bot_deliveries"),
        F.countDistinct("user_id").alias("active_users"),
    ).collect()[0]
    late = d.filter((F.col("truth_delivery") == 0) & (F.to_date("truth_ts") < F.to_date("ingest_ts"))).count()
    per_batch = {int(r["batch_id"]): int(r["count"]) for r in d.groupBy("batch_id").count().collect()}
    info = {
        "scale": sc.name,
        "seed": sc.seed,
        "config": repr(sc),
        "deliveries": int(agg["deliveries"]),
        "unique_events": int(agg["deliveries"]) - int(agg["duplicate_deliveries"]),
        "duplicate_deliveries": int(agg["duplicate_deliveries"]),
        "bot_deliveries": int(agg["bot_deliveries"]),
        "late_events_next_day": int(late),
        "active_users": int(agg["active_users"]),
        "deliveries_per_batch": dict(sorted(per_batch.items())),
        "generate_seconds": round(time.perf_counter() - t0, 1),
    }
    marker.write_text(json.dumps(info, indent=2))
    return info


def write_sample(spark: SparkSession, sc: Scale, path: Path) -> int:
    """Committed SYNTHETIC sample (gzip CSV, sorted, <= 1 MB) from the tiny TEST preset."""
    df = raw_deliveries(spark, sc).orderBy("batch_id", "ingest_ts", "event_id")
    pdf = df.toPandas()
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return len(pdf)
