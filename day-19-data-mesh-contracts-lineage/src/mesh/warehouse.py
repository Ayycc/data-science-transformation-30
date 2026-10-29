"""Local warehouse: one DuckDB file per domain (each domain owns its storage) plus one for the consumer.

* ``load_raw`` lands each domain's raw CSV files verbatim in ``<domain>.duckdb`` schema ``raw``
  (``union_by_name`` so every historical raw schema version is kept side by side).
* ``render_contract_yaml`` regenerates dbt model-contract YAML from the ODCS contracts.
* ``dbt_build`` runs ``dbt build`` for one project in-process via ``dbtRunner``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import duckdb

from mesh import contracts as C
from mesh.config import DBT, DOMAINS, RAW, WAREHOUSE

RAW_TABLE = {"orders": "orders", "payments": "payments", "logistics": "shipments"}


def raw_glob(domain: str, raw_dir: Path = RAW) -> str:
    return str(raw_dir / domain / f"{domain}_*.csv")


def load_raw(raw_dir: Path = RAW, warehouse: Path = WAREHOUSE) -> dict[str, dict[str, Any]]:
    warehouse.mkdir(parents=True, exist_ok=True)
    out = {}
    for domain in DOMAINS:
        db = warehouse / f"{domain}.duckdb"
        if db.exists():
            db.unlink()
        con = duckdb.connect(str(db))
        con.execute("create schema if not exists raw")
        tbl = RAW_TABLE[domain]
        con.execute(
            f"create or replace table raw.{tbl} as select * exclude (filename), filename as _source_file "
            f"from read_csv('{raw_glob(domain, raw_dir)}', union_by_name = true, filename = true, header = true)"
        )
        cols = [
            r[0]
            for r in con.execute(
                f"select column_name from information_schema.columns where table_name = '{tbl}' order by ordinal_position"
            ).fetchall()
        ]
        n = con.execute(f"select count(*) from raw.{tbl}").fetchone()[0]
        files = con.execute(f"select count(distinct _source_file) from raw.{tbl}").fetchone()[0]
        con.close()
        out[domain] = {"table": f"{domain}.raw.{tbl}", "rows": int(n), "files": int(files), "columns": cols}
    return out


def render_contract_yaml() -> dict[str, str]:
    paths = {}
    for domain in DOMAINS:
        c = C.current(domain)
        errs = C.validate(c)
        if errs:
            raise ValueError(f"{domain} contract invalid: {errs}")
        p = C.dbt_yaml_path(domain)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(C.render_dbt_yaml(c))
        paths[domain] = str(p.relative_to(DBT.parent))
    return paths


def dbt_build(
    project: str,
    warehouse: Path = WAREHOUSE,
    select: str | None = None,
    project_dir: Path | None = None,
    target_path: Path | None = None,
) -> dict[str, Any]:
    """Run ``dbt build`` for one project. Returns success flag, node statuses and the manifest path."""
    from dbt.cli.main import dbtRunner

    pdir = project_dir or (DBT / project)
    tdir = target_path or (pdir / "target")
    os.environ["MESH_WAREHOUSE"] = str(warehouse.resolve())
    os.environ.setdefault("DBT_SEND_ANONYMOUS_USAGE_STATS", "false")
    args = [
        "build",
        "--project-dir",
        str(pdir),
        "--profiles-dir",
        str(pdir),
        "--log-path",
        str(tdir.parent / "logs"),
        "--target-path",
        str(tdir),
        "--quiet",
    ]
    if select:
        args += ["--select", select]
    res = dbtRunner().invoke(args)
    statuses: list[dict[str, Any]] = []
    if res.result is not None and hasattr(res.result, "results"):
        for r in res.result.results:
            statuses.append({"node": r.node.unique_id, "status": str(r.status), "message": (r.message or "")[:500]})
    return {
        "project": project,
        "success": bool(res.success),
        "results": statuses,
        "manifest": str(tdir / "manifest.json"),
        "exception": None if res.exception is None else str(res.exception)[:1000],
    }


def manifest(project: str, target_path: Path | None = None) -> dict[str, Any]:
    return json.loads(((target_path or DBT / project / "target") / "manifest.json").read_text())


def connect_ro(path: Path) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB file for reading. dbt (run in-process) may still hold the file with another
    configuration; DuckDB then refuses a read-only handle, so fall back to the default one."""
    try:
        return duckdb.connect(str(path), read_only=True)
    except duckdb.ConnectionException:
        return duckdb.connect(str(path))


def catalog_schema(warehouse: Path = WAREHOUSE) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    """{database: {schema: {table: {column: type}}}} across every DuckDB file - used to qualify SQL for lineage."""
    out: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for f in sorted(warehouse.glob("*.duckdb")):
        con = connect_ro(f)
        rows = con.execute(
            "select table_catalog, table_schema, table_name, column_name, data_type from information_schema.columns "
            "where table_catalog = current_database() order by table_name, ordinal_position"
        ).fetchall()
        con.close()
        for db, sch, tbl, col, typ in rows:
            out.setdefault(db, {}).setdefault(sch, {}).setdefault(tbl, {})[col] = typ
    return out
