"""End-to-end orchestration: bronze load -> SCD2 snapshot replay -> dbt build (two phases).

Phase 1 builds everything against a *fresh* DuckDB file with ingestion batches up to
`PHASE1_LAST_BATCH`. Phase 2 lands the remaining batches (which include late-arriving
lines for earlier months and returns whose lines were still missing) and re-runs
`dbt build`, exercising the incremental silver model.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import duckdb

from retail_lakehouse.bronze import load_bronze
from retail_lakehouse.config import DBT_DIR, PHASE1_LAST_BATCH

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("DBT_SEND_ANONYMOUS_USAGE_STATS", "false")


class DbtError(RuntimeError):
    """Raised when a dbt invocation fails."""


def _release_duckdb() -> None:
    """Drop dbt-duckdb's cached connection so Python code can reopen the database file."""
    import gc

    from dbt.adapters.duckdb.connections import DuckDBConnectionManager

    DuckDBConnectionManager.close_all_connections()
    gc.collect()


def run_dbt(args: list[str], db_path: Path, target_dir: Path, dbt_vars: dict | None = None) -> dict:
    """Invoke dbt in-process and return a summary of node results."""
    from dbt.cli.main import dbtRunner

    os.environ["DUCKDB_PATH"] = str(Path(db_path).resolve())
    cli = [
        *args,
        "--project-dir", str(DBT_DIR),
        "--profiles-dir", str(DBT_DIR),
        "--target-path", str(target_dir),
        "--log-path", str(Path(target_dir) / "logs"),
        "--quiet",
    ]  # fmt: skip
    if dbt_vars:
        cli += ["--vars", json.dumps(dbt_vars)]
    try:
        res = dbtRunner().invoke(cli)
    finally:
        _release_duckdb()
    summary: dict = {"command": " ".join(args), "success": bool(res.success)}
    results = getattr(res.result, "results", None)
    if results is not None:
        counts: Counter = Counter()
        failures = []
        for r in results:
            rtype = getattr(r.node, "resource_type", "unknown")
            rtype = getattr(rtype, "value", str(rtype))
            status = getattr(r.status, "value", str(r.status))
            counts[f"{rtype}:{status}"] += 1
            if status not in ("success", "pass", "no-op"):
                failures.append({"node": r.node.unique_id, "status": status, "message": r.message})
        summary["counts"] = dict(sorted(counts.items()))
        summary["failures"] = failures
    if not res.success:
        detail = summary.get("failures") or repr(res.exception)
        raise DbtError(f"dbt {' '.join(args)} failed: {detail}")
    return summary


def _extract_dates(db_path: Path) -> list[str]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "select distinct cast(extract_date as date) from bronze.product_catalog order by 1"
        ).fetchall()
    finally:
        con.close()
    return [r[0].isoformat() for r in rows]


def _phase_stats(db_path: Path, cutoff: str) -> dict:
    """Headline state of gold after a build; `cutoff` = first day of the phase-2 batches."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        q = f"""
            select
                (select count(*) from gold.fct_order_lines) as fct_order_lines,
                (select sum(net_amount) from gold.fct_order_lines) as net_revenue,
                (select sum(net_amount) from gold.fct_order_lines
                    where order_ts_utc < timestamp '{cutoff}') as net_revenue_before_cutoff,
                (select count(*) from gold.fct_order_lines
                    where order_ts_utc < timestamp '{cutoff}') as lines_before_cutoff,
                (select count(*) from gold.fct_returns) as fct_returns,
                (select count(*) from silver.int_returns__unmatched) as returns_awaiting_order_line,
                (select max(last_load_id) from silver.int_order_lines__deduped) as max_load_id
        """
        row = con.execute(q).fetchone()
        cols = [d[0] for d in con.description]
    finally:
        con.close()
    return {c: (float(v) if hasattr(v, "as_tuple") else v) for c, v in zip(cols, row)}


def run_pipeline(
    landing_dir: Path,
    db_path: Path,
    target_dir: Path,
    phase1_last_batch: str = PHASE1_LAST_BATCH,
    docs: bool = True,
) -> dict:
    """Run the full medallion build from a landing zone into a fresh DuckDB file."""
    db_path, target_dir = Path(db_path), Path(target_dir)
    for p in (db_path, Path(f"{db_path}.wal")):
        if p.exists():
            p.unlink()
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    def lap(name: str, start: float) -> None:
        timings[name] = round(time.perf_counter() - start, 2)

    s = time.perf_counter()
    load1 = load_bronze(db_path, landing_dir, max_batch=phase1_last_batch)
    lap("phase1_bronze_load_s", s)

    s = time.perf_counter()
    seed = run_dbt(["seed"], db_path, target_dir)
    extracts = _extract_dates(db_path)
    for d in extracts:
        run_dbt(["snapshot"], db_path, target_dir, {"catalog_extract_date": d})
    lap("snapshot_replay_s", s)

    s = time.perf_counter()
    build1 = run_dbt(["build"], db_path, target_dir)
    lap("phase1_dbt_build_s", s)
    y, m = (int(x) for x in phase1_last_batch.split("-"))
    cutoff = f"{y + m // 12}-{m % 12 + 1:02d}-01"
    stats1 = _phase_stats(db_path, cutoff)

    s = time.perf_counter()
    load2 = load_bronze(db_path, landing_dir, max_batch=None)
    lap("phase2_bronze_load_s", s)

    s = time.perf_counter()
    build2 = run_dbt(["build"], db_path, target_dir)
    lap("phase2_dbt_build_s", s)
    stats2 = _phase_stats(db_path, cutoff)

    docs_summary = None
    if docs:
        s = time.perf_counter()
        docs_summary = run_dbt(["docs", "generate"], db_path, target_dir)
        lap("dbt_docs_generate_s", s)
    timings["pipeline_total_s"] = round(time.perf_counter() - t0, 2)
    return {
        "bronze_loads": [load1, load2],
        "seed": seed,
        "catalog_extracts_replayed": len(extracts),
        "phase2_first_order_date": cutoff,
        "phase1_stats": stats1,
        "phase2_stats": stats2,
        "phase1_build": build1,
        "phase2_build": build2,
        "docs": docs_summary,
        "timings": timings,
    }
