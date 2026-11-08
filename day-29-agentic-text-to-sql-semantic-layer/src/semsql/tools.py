"""Agent tools over the semantic layer. Both the rules planner and the Claude agent call these."""

from __future__ import annotations

import datetime as dt
import decimal
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import duckdb
import sqlglot

from .guardrails import Guardrails, GuardrailViolation, QueryResult, guarded_execute
from .semantic import QuerySpec, SemanticError, SemanticLayer

CATEGORICAL_DIMS = (
    "region",
    "segment",
    "industry",
    "billing_currency",
    "plan",
    "priority",
)
_DIM_SOURCE = {
    "region": ("accounts", "region"),
    "segment": ("accounts", "segment"),
    "industry": ("accounts", "industry"),
    "billing_currency": ("accounts", "billing_currency"),
    "plan": ("subscriptions", "plan"),
    "priority": ("support_tickets", "priority"),
}


def _jsonable(v: Any) -> Any:
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return float(v)
    return v


def sql_cache_key(sql: str) -> str:
    """Cache key = hash of the normalised (re-generated) SQL, so formatting changes still hit."""
    norm = sqlglot.transpile(sql, read="duckdb", write="duckdb", pretty=False)[0]
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


@dataclass
class Toolbox:
    layer: SemanticLayer
    con: duckdb.DuckDBPyConnection
    guard: Guardrails = None  # type: ignore[assignment]
    cache: dict[str, QueryResult] = field(default_factory=dict)
    cache_hits: int = 0
    _compiled: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.guard is None:
            self.guard = Guardrails(self.layer.pii_columns, self.layer.max_rows)
        self._values: dict[str, list[str]] | None = None

    # ---------------------------------------------------------------- catalogue tools
    def list_metrics(self) -> list[dict[str, Any]]:
        return self.layer.list_metrics()

    def describe_metric(self, name: str) -> dict[str, Any]:
        return self.layer.describe_metric(name)

    def list_dimensions(self) -> list[dict[str, Any]]:
        return [self.layer.describe_dimension(d) for d in self.layer.dimensions]

    def dimension_values(self) -> dict[str, list[str]]:
        if self._values is None:
            self._values = {}
            for dim, (table, col) in _DIM_SOURCE.items():
                r = guarded_execute(
                    self.con,
                    self.guard,
                    f"SELECT DISTINCT {col} FROM {table} ORDER BY 1",
                )
                self._values[dim] = [str(x[0]) for x in r.rows if x[0] is not None]
        return self._values

    def describe_dimension(self, name: str) -> dict[str, Any]:
        d = self.layer.describe_dimension(name)
        if name in CATEGORICAL_DIMS:
            d["values"] = self.dimension_values()[name]
        elif name == "fiscal_year":
            d["values"] = [2024, 2025, 2026]
        return d

    # ---------------------------------------------------------------- query tools
    def compile_query(self, spec: QuerySpec | dict[str, Any]) -> dict[str, Any]:
        q = spec if isinstance(spec, QuerySpec) else QuerySpec.from_dict(spec)
        try:
            sql = self.layer.compile(q)
        except (SemanticError, ValueError) as e:
            return {
                "ok": False,
                "error": str(e),
                "code": getattr(e, "code", "invalid_spec"),
            }
        qid = sql_cache_key(sql)
        self._compiled[qid] = sql
        return {
            "ok": True,
            "query_id": qid,
            "sql": sql,
            "caveats": self.layer.caveats(q),
        }

    def run_query(self, query_id: str | None = None, sql: str | None = None) -> dict[str, Any]:
        """Run a compiled query (by id) or raw SQL. Every path goes through the guardrails."""
        if query_id is not None:
            if query_id not in self._compiled:
                return {
                    "ok": False,
                    "error": f"unknown query_id {query_id}; call compile_query first",
                }
            sql = self._compiled[query_id]
        if not sql:
            return {"ok": False, "error": "no SQL given"}
        key = sql_cache_key(sql) if self._parses(sql) else None
        if key and key in self.cache:
            self.cache_hits += 1
            res = self.cache[key]
            return self._pack(res, cached=True)
        try:
            res = guarded_execute(self.con, self.guard, sql)
        except GuardrailViolation as e:
            return {
                "ok": False,
                "blocked": True,
                "error": "blocked by guardrails: " + "; ".join(e.reasons),
            }
        except duckdb.Error as e:
            return {
                "ok": False,
                "error": f"database error: {str(e).splitlines()[0][:300]}",
            }
        if key:
            self.cache[key] = res
        return self._pack(res, cached=False)

    @staticmethod
    def _parses(sql: str) -> bool:
        try:
            sqlglot.transpile(sql, read="duckdb")
            return True
        except sqlglot.errors.SqlglotError:
            return False

    @staticmethod
    def _pack(res: QueryResult, cached: bool) -> dict[str, Any]:
        return {
            "ok": True,
            "columns": res.columns,
            "rows": [[_jsonable(v) for v in r] for r in res.rows],
            "row_count": len(res.rows),
            "truncated": res.truncated,
            "cached": cached,
            "executed_sql": res.sql,
        }

    def check_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Sanity checks an answer must pass before it is reported."""
        warnings: list[str] = []
        if not result.get("ok"):
            return {"ok": False, "warnings": ["query failed"]}
        rows = result["rows"]
        if not rows:
            warnings.append("empty result: check filters and the data range (FY2024-FY2026)")
        if result.get("truncated"):
            warnings.append(f"result truncated at the {self.layer.max_rows}-row limit")
        cols = result["columns"]
        for j, c in enumerate(cols):
            vals = [r[j] for r in rows]
            if c in self.layer.metrics and vals and all(v is None for v in vals):
                warnings.append(f"metric {c} is NULL for every row")
            if c in ("revenue_usd", "arr_usd") and any(isinstance(v, (int, float)) and v < 0 for v in vals):
                warnings.append(f"negative values in {c}")
        return {
            "ok": not any(w.startswith("metric") for w in warnings),
            "warnings": warnings,
        }

    def raw_schema_ddl(self) -> str:
        """Raw schema (no business definitions) for the raw-schema prompting baseline."""
        rows = self.con.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
        ).fetchall()
        out: dict[str, list[str]] = {}
        for t, c, ty in rows:
            out.setdefault(t, []).append(f"{c} {ty}")
        return "\n".join(f"TABLE {t} ({', '.join(cs)})" for t, cs in out.items())


def tool_result_json(obj: Any, max_chars: int = 12000) -> str:
    s = json.dumps(obj, default=_jsonable)
    return s if len(s) <= max_chars else s[:max_chars] + '..."[truncated]"'
