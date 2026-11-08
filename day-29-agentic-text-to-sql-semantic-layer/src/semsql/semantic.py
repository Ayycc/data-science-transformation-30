"""YAML semantic layer: loading, validation and a small QuerySpec -> SQL compiler."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_LAYER = Path(__file__).with_name("semantic_layer.yaml")


class SemanticError(ValueError):
    """Raised when a query cannot be expressed with the semantic layer."""

    code = "semantic_error"


class UndefinedMetricError(SemanticError):
    code = "undefined_metric"


class UndefinedDimensionError(SemanticError):
    code = "undefined_dimension"


class UnsupportedCombinationError(SemanticError):
    code = "unsupported_combination"


@dataclass
class QuerySpec:
    """A structured analytics request: what the planner (rules or LLM) must produce."""

    metrics: list[str]
    dimensions: list[str] = field(default_factory=list)
    filters: dict[str, list[Any]] = field(default_factory=dict)
    as_of: str | None = None
    order_by: list[list[str]] = field(default_factory=list)  # [[column, "asc"|"desc"], ...]
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuerySpec:
        filters = {k: (v if isinstance(v, list) else [v]) for k, v in (d.get("filters") or {}).items()}
        return cls(
            metrics=list(d.get("metrics") or []),
            dimensions=list(d.get("dimensions") or []),
            filters=filters,
            as_of=d.get("as_of"),
            order_by=[list(x) for x in (d.get("order_by") or [])],
            limit=d.get("limit"),
        )


def _lit(v: Any) -> str:
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


class SemanticLayer:
    def __init__(self, path: Path | str = DEFAULT_LAYER):
        with open(path, encoding="utf-8") as f:
            self.spec: dict[str, Any] = yaml.safe_load(f)
        self.models: dict[str, dict] = self.spec["models"]
        self.metrics: dict[str, dict] = self.spec["metrics"]
        self.dimensions: dict[str, dict] = self.spec["dimensions"]
        self.conventions: dict[str, Any] = self.spec["conventions"]
        self.max_rows: int = int(self.spec.get("max_rows", 1000))
        self.pii_columns: list[str] = list(self.spec.get("pii_columns", []))
        self.data_end = str(self.conventions["data_end_date"])
        self._validate()

    # ------------------------------------------------------------------ catalogue
    def _validate(self) -> None:
        for name, m in self.metrics.items():
            if m["model"] not in self.models:
                raise ValueError(f"metric {name} references unknown model {m['model']}")
        for name, d in self.dimensions.items():
            if not d.get("time") and "requires" not in d:
                raise ValueError(f"dimension {name} needs 'requires' or 'time'")

    def available_aliases(self, model_name: str) -> set[str]:
        m = self.models[model_name]
        return {m["alias"], *m.get("joins", {}).keys()}

    def dimension_available(self, metric: str, dim: str) -> bool:
        model_name = self.metrics[metric]["model"]
        model = self.models[model_name]
        d = self.dimensions[dim]
        if d.get("time"):
            return not model.get("snapshot", False)
        return d["requires"] in self.available_aliases(model_name)

    def dimensions_for(self, metric: str) -> list[str]:
        return [d for d in self.dimensions if self.dimension_available(metric, d)]

    def is_snapshot(self, metric: str) -> bool:
        return bool(self.models[self.metrics[metric]["model"]].get("snapshot"))

    def list_metrics(self) -> list[dict[str, Any]]:
        return [
            {
                "name": n,
                "label": m["label"],
                "description": m["description"],
                "point_in_time": self.is_snapshot(n),
            }
            for n, m in self.metrics.items()
        ]

    def describe_metric(self, name: str) -> dict[str, Any]:
        if name not in self.metrics:
            raise UndefinedMetricError(f"metric '{name}' is not defined in the semantic layer")
        m = self.metrics[name]
        return {
            "name": name,
            "label": m["label"],
            "description": m["description"],
            "point_in_time": self.is_snapshot(name),
            "dimensions": self.dimensions_for(name),
        }

    def describe_dimension(self, name: str) -> dict[str, Any]:
        if name not in self.dimensions:
            raise UndefinedDimensionError(f"dimension '{name}' is not defined in the semantic layer")
        d = self.dimensions[name]
        return {
            "name": name,
            "description": d["description"],
            "time": bool(d.get("time")),
            "metrics": [m for m in self.metrics if self.dimension_available(m, name)],
        }

    # ------------------------------------------------------------------ compiler
    def validate_spec(self, q: QuerySpec) -> None:
        if not q.metrics:
            raise UndefinedMetricError("no metric requested")
        for m in q.metrics:
            if m not in self.metrics:
                raise UndefinedMetricError(f"metric '{m}' is not defined in the semantic layer")
        for d in [*q.dimensions, *q.filters]:
            if d not in self.dimensions:
                raise UndefinedDimensionError(f"dimension '{d}' is not defined in the semantic layer")
        for m in q.metrics:
            for d in [*q.dimensions, *q.filters]:
                if not self.dimension_available(m, d):
                    why = (
                        f"'{m}' is a point-in-time metric; use as_of instead of the time dimension '{d}'"
                        if self.is_snapshot(m) and self.dimensions[d].get("time")
                        else f"dimension '{d}' cannot be joined to metric '{m}'"
                    )
                    raise UnsupportedCombinationError(why)
        if q.as_of is not None:
            dt.date.fromisoformat(q.as_of)  # raises ValueError if malformed
            if not any(self.is_snapshot(m) for m in q.metrics):
                raise UnsupportedCombinationError("as_of only applies to point-in-time metrics")
        allowed_cols = set(q.metrics) | set(q.dimensions)
        for col, direction in q.order_by:
            if col not in allowed_cols or direction not in ("asc", "desc"):
                raise SemanticError(f"invalid order_by {col} {direction}")
        if q.limit is not None and (not isinstance(q.limit, int) or q.limit < 1):
            raise SemanticError("limit must be a positive integer")

    def compile(self, q: QuerySpec) -> str:
        """Compile a QuerySpec into a single read-only DuckDB SELECT."""
        self.validate_spec(q)
        as_of = q.as_of or self.data_end
        by_model: dict[str, list[str]] = {}
        for m in q.metrics:
            by_model.setdefault(self.metrics[m]["model"], []).append(m)
        ctes = []
        for i, (model_name, metrics) in enumerate(by_model.items()):
            ctes.append(
                (
                    f"m{i}_{model_name}",
                    self._compile_model(model_name, metrics, q, as_of),
                )
            )
        dims = q.dimensions
        if len(ctes) == 1:
            from_sql = ctes[0][0]
        else:
            from_sql = ctes[0][0]
            for name, _ in ctes[1:]:
                from_sql += f"\n  FULL OUTER JOIN {name} USING ({', '.join(dims)})" if dims else f"\n  CROSS JOIN {name}"
        select_cols = [*dims, *q.metrics]
        order = q.order_by or [[d, "asc"] for d in dims]
        limit = min(q.limit or self.max_rows, self.max_rows)
        sql = "WITH " + ",\n".join(f"{n} AS (\n{body}\n)" for n, body in ctes)
        sql += f"\nSELECT {', '.join(select_cols)}\nFROM {from_sql}"
        if order:
            sql += "\nORDER BY " + ", ".join(f"{c} {d.upper()} NULLS LAST" for c, d in order)
        sql += f"\nLIMIT {limit}"
        return sql

    def _compile_model(self, model_name: str, metrics: list[str], q: QuerySpec, as_of: str) -> str:
        model = self.models[model_name]
        alias = model["alias"]
        needed = list(model.get("always_join", []))
        uses_time = False
        for d in [*q.dimensions, *q.filters]:
            dd = self.dimensions[d]
            if dd.get("time"):
                uses_time = True
            elif dd["requires"] != alias and dd["requires"] not in needed:
                needed.append(dd["requires"])
        lines = ["  SELECT"]
        sel = [f"{self.dimensions[d]['sql']} AS {d}" for d in q.dimensions]
        sel += [f"{self.metrics[m]['sql']} AS {m}" for m in metrics]
        lines.append("    " + ",\n    ".join(sel))
        lines.append(f"  FROM {model['from']} {alias}")
        for j in needed:
            lines.append("  " + model["joins"][j].replace("{as_of}", as_of))
        if uses_time:
            lines.append(f"  JOIN dim_date d ON d.date_day = {model['time_column']}")
        where = [w.replace("{as_of}", as_of) for w in model.get("where", [])]
        for d, values in q.filters.items():
            col = self.dimensions[d]["sql"]
            vals = ", ".join(_lit(v) for v in values)
            where.append(f"{col} IN ({vals})")
        if where:
            lines.append("  WHERE " + "\n    AND ".join(where))
        if q.dimensions:
            lines.append("  GROUP BY " + ", ".join(self.dimensions[d]["sql"] for d in q.dimensions))
        return "\n".join(lines)

    def caveats(self, q: QuerySpec) -> list[str]:
        out = [f"{self.metrics[m]['label']}: {self.metrics[m]['description']}" for m in q.metrics]
        if any(self.is_snapshot(m) for m in q.metrics):
            out.append(f"Point-in-time metric evaluated as of {q.as_of or self.data_end}.")
        if any(self.dimensions[d].get("time") and d.startswith("fiscal") for d in [*q.dimensions, *q.filters]):
            out.append("Fiscal years start on 1 February and are named after the year they end (FY2025 = Feb 2024-Jan 2025).")
        if "calendar_year" in q.filters or "calendar_month" in q.dimensions:
            out.append("Calendar periods (January-December), not fiscal.")
        out.append("Test accounts are excluded. Data is SYNTHETIC.")
        return out
