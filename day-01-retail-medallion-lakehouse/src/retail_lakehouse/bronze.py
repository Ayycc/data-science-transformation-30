"""Bronze layer: land raw Parquet files into DuckDB `bronze.*` tables, untouched.

Event feeds (POS lines, web lines, returns) are appended batch by batch and every
row is stamped with `_source_file` and a monotonically increasing `_load_id`, which
the silver incremental model uses as its watermark. Master data (stores, customers,
catalog extracts) is replaced in full on each load, as the source systems export it.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

EVENT_FEEDS = ("pos_order_lines", "web_order_lines", "order_returns")
MASTER_FILES = {"stores": "stores.parquet", "customers": "customers.parquet",
                "product_catalog": "product_catalog.parquet"}  # fmt: skip


def _batch_of(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]  # e.g. pos_order_lines_2025-03 -> 2025-03


def load_bronze(db_path: Path, landing_dir: Path, max_batch: str | None = None) -> dict:
    """Load landing files into bronze. Only batches <= max_batch (YYYY-MM) when given.

    Returns a dict describing what this load added.
    """
    landing_dir = Path(landing_dir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("create schema if not exists bronze")
        con.execute(
            "create table if not exists bronze._load_log (load_id integer, feed varchar, "
            "source_file varchar, batch varchar, row_count bigint, loaded_at timestamp)"
        )
        load_id = con.execute("select coalesce(max(load_id), 0) + 1 from bronze._load_log").fetchone()[0]
        added: dict[str, dict] = {}

        for table, fname in MASTER_FILES.items():
            src = (landing_dir / fname).as_posix()
            con.execute(
                f"create or replace table bronze.{table} as select *, '{fname}' as _source_file, "
                f"{load_id} as _load_id from read_parquet('{src}')"
            )
            n = con.execute(f"select count(*) from bronze.{table}").fetchone()[0]
            con.execute("insert into bronze._load_log values (?, ?, ?, null, ?, now())", [load_id, table, fname, n])
            added[table] = {"rows": int(n), "files": 1}

        for feed in EVENT_FEEDS:
            files = sorted((landing_dir / feed).glob(f"{feed}_*.parquet"))
            if max_batch is not None:
                files = [f for f in files if _batch_of(f) <= max_batch]
            done = {
                r[0] for r in con.execute("select source_file from bronze._load_log where feed = ?", [feed]).fetchall()
            }
            todo = [f for f in files if f.name not in done]
            if not files:
                raise FileNotFoundError(f"no landing files for {feed} in {landing_dir}")
            con.execute(
                f"create table if not exists bronze.{feed} as select *, ''::varchar as _source_file, "
                f"0 as _load_id from read_parquet('{files[0].as_posix()}') limit 0"
            )
            rows = 0
            for f in todo:
                n = con.execute(
                    f"insert into bronze.{feed} by name select *, '{f.name}' as _source_file, "
                    f"{load_id} as _load_id from read_parquet('{f.as_posix()}')"
                ).fetchone()[0]
                con.execute(
                    "insert into bronze._load_log values (?, ?, ?, ?, ?, now())",
                    [load_id, feed, f.name, _batch_of(f), n],
                )
                rows += n
            added[feed] = {"rows": int(rows), "files": len(todo)}
        return {"load_id": int(load_id), "max_batch": max_batch, "added": added}
    finally:
        con.close()
