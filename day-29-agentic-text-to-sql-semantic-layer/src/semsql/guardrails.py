"""SQL guardrails: every statement passes through here before it touches DuckDB.

Layers (defence in depth):
1. sqlglot parse (DuckDB dialect); unparseable SQL is rejected.
2. Exactly one statement, and it must be a read-only query (SELECT / UNION / ... with CTEs);
   any DML, DDL, COPY, ATTACH, PRAGMA, SET, INSTALL/LOAD or other command node is rejected,
   anywhere in the tree.
3. File / network table functions (read_csv, read_parquet, glob, ...) are rejected.
4. PII columns are rejected by name, and SELECT * is rejected on tables holding PII.
5. Only known warehouse tables may be referenced.
6. The row limit is enforced by rewriting LIMIT (added if missing, capped if too large).
7. Execution happens on a DuckDB connection opened read_only with external access disabled
   and the configuration locked, so a bypass of 1-6 still cannot write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

READ_ONLY_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)
FORBIDDEN_NODES = tuple(
    getattr(exp, n)
    for n in (
        "Insert",
        "Update",
        "Delete",
        "Merge",
        "Create",
        "Drop",
        "Alter",
        "TruncateTable",
        "Command",
        "Copy",
        "Attach",
        "Detach",
        "Pragma",
        "Set",
        "Use",
        "Grant",
        "Revoke",
        "Transaction",
        "Commit",
        "Rollback",
        "Install",
        "Load",
        "LoadData",
        "Export",
        "Into",
        "Analyze",
        "Describe",
        "Summarize",
    )
    if hasattr(exp, n)
)
FORBIDDEN_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "read_json",
    "read_json_auto",
    "read_text",
    "read_blob",
    "parquet_scan",
    "csv_scan",
    "glob",
    "sniff_csv",
    "read_xlsx",
    "query",
    "query_table",
    "sqlite_scan",
    "postgres_scan",
    "mysql_scan",
    "iceberg_scan",
    "delta_scan",
    "read_ndjson",
    "read_ndjson_auto",
    "getenv",
    "duckdb_secrets",
    "duckdb_settings",
    "duckdb_extensions",
    "system",
    "shell",
}
ALLOWED_TABLES = {
    "accounts",
    "subscriptions",
    "invoices",
    "fx_rates",
    "usage_monthly",
    "support_tickets",
    "dim_date",
}


class GuardrailViolation(Exception):
    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass
class GuardResult:
    ok: bool
    sql: str | None
    reasons: list[str] = field(default_factory=list)
    limit_applied: int | None = None


@dataclass
class Guardrails:
    pii_columns: list[str]
    max_rows: int = 1000
    allowed_tables: set[str] = field(default_factory=lambda: set(ALLOWED_TABLES))

    @property
    def pii_names(self) -> set[str]:
        return {c.split(".")[-1].lower() for c in self.pii_columns}

    @property
    def pii_tables(self) -> set[str]:
        return {c.split(".")[0].lower() for c in self.pii_columns if "." in c}

    def check(self, sql: str) -> GuardResult:
        reasons: list[str] = []
        try:
            stmts = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
        except sqlglot.errors.SqlglotError as e:  # ParseError, TokenError
            return GuardResult(False, None, [f"unparseable SQL: {str(e).splitlines()[0][:200]}"])
        if len(stmts) != 1:
            return GuardResult(False, None, [f"exactly one statement allowed, got {len(stmts)}"])
        tree = stmts[0]
        if not isinstance(tree, READ_ONLY_ROOTS):
            reasons.append(f"only read-only SELECT queries are allowed (got {type(tree).__name__.upper()})")
        for node in tree.walk():
            if isinstance(node, FORBIDDEN_NODES):
                reasons.append(f"forbidden statement/clause: {type(node).__name__.upper()}")
                break
        cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
        for fn in tree.find_all(exp.Func):
            name = (fn.sql_name() if not isinstance(fn, exp.Anonymous) else fn.name).lower()
            if name in FORBIDDEN_FUNCTIONS:
                reasons.append(f"forbidden function: {name}")
        for tbl in tree.find_all(exp.Table):
            name = tbl.name.lower()
            if isinstance(tbl.this, exp.Func) or not name:
                # table-valued function in FROM, e.g. FROM read_csv('x')
                fname = tbl.this.sql_name().lower() if isinstance(tbl.this, exp.Func) else "?"
                reasons.append(f"table functions are not allowed: {fname}")
                continue
            if tbl.args.get("db") or tbl.args.get("catalog"):
                reasons.append(f"qualified/external table reference not allowed: {tbl.sql()}")
            elif name not in self.allowed_tables and name not in cte_names:
                reasons.append(f"unknown table: {name}")
        for col in tree.find_all(exp.Column):
            if col.name.lower() in self.pii_names:
                reasons.append(f"PII column blocked: {col.name}")
        for ident in tree.find_all(exp.Identifier):
            if ident.name.lower() in self.pii_names and not any(r.startswith("PII column blocked") for r in reasons):
                reasons.append(f"PII column blocked: {ident.name}")
        star_projection = any(
            isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star))
            for sel in tree.find_all(exp.Select)
            for e in sel.expressions
        )
        if star_projection:
            tables = {t.name.lower() for t in tree.find_all(exp.Table)}
            if tables & self.pii_tables:
                reasons.append("SELECT * is not allowed on tables containing PII columns")
        if reasons:
            return GuardResult(False, None, sorted(set(reasons), key=reasons.index))
        limit_applied = self.max_rows
        lim = tree.args.get("limit")
        if lim is not None:
            try:
                n = int(lim.expression.name) if lim.expression is not None else None
            except (TypeError, ValueError):
                n = None
            if n is None:
                return GuardResult(False, None, ["LIMIT must be a literal integer"])
            limit_applied = min(n, self.max_rows)
        tree = (
            tree.limit(limit_applied, copy=True)
            if isinstance(tree, exp.Select)
            else exp.select("*").from_(tree.subquery("q")).limit(limit_applied)
        )
        return GuardResult(True, tree.sql(dialect="duckdb"), [], limit_applied)


def open_readonly(path: Path | str) -> duckdb.DuckDBPyConnection:
    """Read-only connection with external file/network access disabled and settings locked."""
    con = duckdb.connect(
        str(path),
        read_only=True,
        config={
            "enable_external_access": False,
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
        },
    )
    con.execute("SET lock_configuration = true")
    return con


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool
    sql: str


def guarded_execute(con: duckdb.DuckDBPyConnection, guard: Guardrails, sql: str) -> QueryResult:
    res = guard.check(sql)
    if not res.ok:
        raise GuardrailViolation(res.reasons)
    cur = con.execute(res.sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    truncated = len(rows) >= (res.limit_applied or guard.max_rows) and (res.limit_applied or 0) >= guard.max_rows
    return QueryResult(cols, rows, truncated, res.sql)
