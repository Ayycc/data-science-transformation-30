"""Bronze -> silver -> gold Delta tables with incremental, idempotent MERGEs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from delta.tables import DeltaTable
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from clickstream import transforms as T
from clickstream.generate import landing_batch

BRONZE_DDL = [
    ("event_id", "STRING", False),
    ("user_id", "BIGINT", False),
    ("article_id", "BIGINT", False),
    ("event_type", "STRING", False),
    ("client_ts", "TIMESTAMP", False),
    ("ingest_ts", "TIMESTAMP", False),
    ("device", "STRING", True),
    ("user_agent", "STRING", True),
    ("scroll_depth", "INT", True),
    ("batch_id", "INT", False),
]
SILVER_DDL = [c for c in BRONZE_DDL if c[0] != "batch_id"] + [("event_date", "DATE", False), ("first_batch_id", "INT", False)]
SESSION_DDL = [
    ("session_id", "STRING", False),
    ("user_id", "BIGINT", False),
    ("session_date", "DATE", False),
    ("session_start", "TIMESTAMP", False),
    ("session_end", "TIMESTAMP", False),
    ("duration_s", "DOUBLE", False),
    ("n_events", "BIGINT", False),
    ("n_page_views", "BIGINT", False),
    ("n_scrolls", "BIGINT", False),
    ("n_clicks", "BIGINT", False),
    ("n_ad_impressions", "BIGINT", False),
    ("max_scroll_depth", "INT", True),
    ("n_articles", "BIGINT", False),
    ("landing_article_id", "BIGINT", False),
    ("exit_article_id", "BIGINT", False),
    ("device", "STRING", True),
    ("is_bot", "BOOLEAN", False),
    ("row_hash", "BIGINT", False),
]
PROFILE_DDL = [
    ("user_id", "BIGINT", False),
    ("clock_offset_h", "INT", False),
    ("is_bot", "BOOLEAN", False),
    ("bot_reason", "STRING", True),
    ("max_events_per_day", "BIGINT", False),
]
EVENT_TYPES = ("page_view", "scroll", "click", "ad_impression")


def _create(spark: SparkSession, path: str, cols: list[tuple[str, str, bool]], partition: str | None, comment: str) -> None:
    b = DeltaTable.createIfNotExists(spark).location(path).comment(comment)
    for name, typ, nullable in cols:
        b = b.addColumn(name, typ, nullable=nullable)
    if partition:
        b = b.partitionedBy(partition)
    b.execute()


@dataclass
class Lakehouse:
    spark: SparkSession
    root: Path
    deliveries_path: str  # landing zone (parquet, partitioned by batch_id)
    last_batch: int
    _cached: list = field(default_factory=list)

    def p(self, name: str) -> str:
        return str((self.root / name).resolve())

    @property
    def bronze(self) -> str:
        return self.p("bronze_events")

    @property
    def silver(self) -> str:
        return self.p("silver_events")

    @property
    def sessions(self) -> str:
        return self.p("gold_sessions")

    @property
    def profiles(self) -> str:
        return self.p("gold_user_profiles")

    @property
    def article_daily(self) -> str:
        return self.p("gold_article_daily")

    # ------------------------------------------------------------------ setup
    def create_tables(self) -> None:
        s = self.spark
        _create(s, self.bronze, BRONZE_DDL, "batch_id", "SYNTHETIC raw click deliveries, append-only")
        if not self._has_constraint(self.bronze, "valid_event_type"):
            types = ", ".join(f"'{t}'" for t in EVENT_TYPES)
            s.sql(f"ALTER TABLE delta.`{self.bronze}` ADD CONSTRAINT valid_event_type CHECK (event_type IN ({types}))")
        _create(s, self.silver, SILVER_DDL, "event_date", "SYNTHETIC deduplicated events, one row per event_id")
        _create(s, self.sessions, SESSION_DDL, "session_date", "SYNTHETIC 30-minute-inactivity sessions")
        _create(s, self.profiles, PROFILE_DDL, None, "SYNTHETIC per-user clock offset and bot flag")

    def _has_constraint(self, path: str, name: str) -> bool:
        props = self.spark.sql(f"SHOW TBLPROPERTIES delta.`{path}`").collect()
        return any(r["key"] == f"delta.constraints.{name}" for r in props)

    def landing(self, batch_id: int) -> DataFrame:
        raw = self.spark.read.parquet(self.deliveries_path)
        return landing_batch(raw, batch_id, self.last_batch)

    # ------------------------------------------------------------------ bronze
    def ingest_bronze(self, batch_id: int, df: DataFrame | None = None) -> None:
        """Append one landing batch. ``txnAppId/txnVersion`` make the append idempotent: a rerun is a no-op."""
        df = self.landing(batch_id) if df is None else df
        df = df.withColumn("batch_id", F.lit(batch_id))
        (
            df.write.format("delta")
            .mode("append")
            .option("mergeSchema", "true")  # schema evolution: new nullable columns (app_version) are added
            .option("txnAppId", "clickstream-bronze")
            .option("txnVersion", batch_id)
            .save(self.bronze)
        )

    # ------------------------------------------------------------------ silver
    def merge_silver(self, batch_id: int) -> None:
        """Deduplicate a bronze batch and MERGE it into silver on event_id.

        Insert-only MERGE (Delta runs it as an anti-join + append): a redelivery of an event already in silver
        always has a later ingest_ts than the stored copy, because batches are ingest days, so it is simply dropped.
        """
        src = self.spark.read.format("delta").load(self.bronze).filter(F.col("batch_id") == batch_id)
        src = T.dedup_deliveries(src.drop("batch_id")).withColumn("event_date", F.to_date("client_ts"))
        src = src.withColumn("first_batch_id", F.lit(batch_id))
        dates = sorted(r[0] for r in src.select("event_date").distinct().collect())
        date_list = ", ".join(f"DATE'{d}'" for d in dates) or "NULL"
        cond = f"t.event_date IN ({date_list}) AND t.event_date = s.event_date AND t.event_id = s.event_id"
        (
            DeltaTable.forPath(self.spark, self.silver)
            .alias("t")
            .merge(src.alias("s"), cond)
            .withSchemaEvolution()
            .whenNotMatchedInsertAll()
            .execute()
        )

    # ------------------------------------------------------------------ gold
    def recompute_users(self, users: DataFrame) -> tuple[DataFrame, DataFrame]:
        """Rebuild profiles and sessions for ``users`` from their full silver history."""
        silver = self.spark.read.format("delta").load(self.silver)
        ev = silver.join(users, "user_id", "left_semi").select(
            "event_id", "user_id", "article_id", "event_type", "client_ts", "ingest_ts", "device", "user_agent", "scroll_depth"
        )
        offsets = T.estimate_clock_offset(ev)
        ev2 = T.apply_clock_offset(ev, offsets).persist(StorageLevel.MEMORY_AND_DISK)
        self._cached.append(ev2)
        bots = T.bot_flags(ev2)
        profiles = ev2.select("user_id", "clock_offset_h").distinct().join(bots, "user_id").select(*[c[0] for c in PROFILE_DDL])
        profiles = profiles.persist(StorageLevel.MEMORY_AND_DISK)
        self._cached.append(profiles)
        sess = T.session_table(T.sessionise(ev2)).join(profiles.select("user_id", "is_bot"), "user_id")
        sess = sess.withColumn("row_hash", F.xxhash64(*[F.col(c) for c in T.SESSION_COLS + ["is_bot"]]))
        return profiles, sess.select(*[c[0] for c in SESSION_DDL])

    def merge_gold(self, batch_id: int) -> dict:
        """Recompute gold for the users touched by ``batch_id`` and MERGE the change set (upsert + delete)."""
        s = self.spark
        affected = s.read.format("delta").load(self.bronze).filter(F.col("batch_id") == batch_id).select("user_id").distinct()
        affected = affected.persist()
        n_aff = affected.count()
        profiles, sess = self.recompute_users(affected)
        sess = sess.persist(StorageLevel.MEMORY_AND_DISK)
        self._cached.append(sess)
        # Change sets only carry rows that differ from what gold already holds, so a replayed batch
        # rewrites no files (MERGE would otherwise copy every file that contains a matched row).
        pcols = [c[0] for c in PROFILE_DDL]
        current = s.read.format("delta").load(self.profiles).select("user_id", F.xxhash64(*pcols).alias("_h"))
        changed_profiles = profiles.withColumn("_h", F.xxhash64(*pcols)).join(current, ["user_id", "_h"], "left_anti").drop("_h")
        (
            DeltaTable.forPath(s, self.profiles)
            .alias("t")
            .merge(changed_profiles.alias("s"), "t.user_id = s.user_id")
            .whenMatchedUpdateAll(
                condition="NOT (t.clock_offset_h <=> s.clock_offset_h AND t.is_bot <=> s.is_bot "
                "AND t.bot_reason <=> s.bot_reason AND t.max_events_per_day <=> s.max_events_per_day)"
            )
            .whenNotMatchedInsertAll()
            .execute()
        )
        old = s.read.format("delta").load(self.sessions).join(affected, "user_id", "left_semi")
        old = old.select("session_id", "session_date", "row_hash").persist(StorageLevel.MEMORY_AND_DISK)
        self._cached.append(old)
        deletes = old.drop("row_hash").join(sess.select("session_id"), "session_id", "left_anti").withColumn("_action", F.lit("delete"))
        ups = sess.join(old.select("session_id", "row_hash"), ["session_id", "row_hash"], "left_anti").withColumn(
            "_action", F.lit("upsert")
        )
        change = ups.unionByName(deletes, allowMissingColumns=True)
        cols = [c[0] for c in SESSION_DDL]
        (
            DeltaTable.forPath(s, self.sessions)
            .alias("t")
            .merge(change.alias("s"), "t.session_date = s.session_date AND t.session_id = s.session_id")
            .whenMatchedDelete(condition="s._action = 'delete'")
            .whenMatchedUpdate(condition="s._action = 'upsert' AND t.row_hash <> s.row_hash", set={c: f"s.{c}" for c in cols})
            .whenNotMatchedInsert(condition="s._action = 'upsert'", values={c: f"s.{c}" for c in cols})
            .execute()
        )
        self.release()
        affected.unpersist()
        self.rebuild_article_daily()
        return {"affected_users": n_aff}

    def rebuild_article_daily(self) -> None:
        """Article x day human engagement. Small aggregate, rebuilt in full (deterministic, so idempotent)."""
        s = self.spark
        silver = s.read.format("delta").load(self.silver)
        prof = s.read.format("delta").load(self.profiles)
        out = T.article_daily(silver, prof)
        out.write.format("delta").mode("overwrite").partitionBy("event_date").option("overwriteSchema", "true").save(self.article_daily)

    # ------------------------------------------------------------------ helpers
    def release(self) -> None:
        for df in self._cached:
            df.unpersist()
        self._cached.clear()

    def run_batch(self, batch_id: int) -> dict:
        t = {}
        t0 = time.perf_counter()
        self.ingest_bronze(batch_id)
        t["bronze_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.merge_silver(batch_id)
        t["silver_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        info = self.merge_gold(batch_id)
        t["gold_s"] = time.perf_counter() - t0
        return {**info, **{k: round(v, 2) for k, v in t.items()}}

    def version(self, path: str) -> int:
        return int(DeltaTable.forPath(self.spark, path).history(1).select("version").collect()[0][0])

    def gold_checksums(self, version: dict[str, int] | None = None) -> dict[str, tuple[int, str]]:
        out = {}
        for name, path in [
            ("gold_sessions", self.sessions),
            ("gold_user_profiles", self.profiles),
            ("gold_article_daily", self.article_daily),
        ]:
            r = self.spark.read.format("delta")
            if version and name in version:
                r = r.option("versionAsOf", version[name])
            out[name] = T.table_checksum(r.load(path))
        return out

    def last_operation_metrics(self, path: str) -> dict:
        h = DeltaTable.forPath(self.spark, path).history(1).collect()[0]
        return {
            "version": int(h["version"]),
            "operation": h["operation"],
            **{k: int(v) for k, v in (h["operationMetrics"] or {}).items() if v.isdigit()},
        }
