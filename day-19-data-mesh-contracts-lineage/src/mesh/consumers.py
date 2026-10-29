"""Replay 90 days of upstream change against downstream dashboards, with and without the mesh.

* **Naive world** - dashboards query each domain's raw feed directly (the pre-mesh status quo), so
  on day ``d`` they see the raw schema of that day (``generators.RAW_RELEASES``).
* **Mesh world** - dashboards query a *published, major-pinned* interface (``fct_orders_v1`` ...).
  On day ``d`` they see the contract version in force that day for that major, either as the latest
  version or as a dual-published older major. A dashboard breaks if its interface is not published
  or lacks a referenced column.

A dashboard is "broken" on a day if any column it references is missing. The check is static (sqlglot
column references against the schema), so it is exactly the "column renamed, dashboard errors" failure
mode from the business brief. Silent semantic changes are not counted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import sqlglot
from sqlglot import exp

from mesh import contracts as C
from mesh.config import N_DAYS, START
from mesh.generators import raw_release_on


@dataclass(frozen=True)
class Dashboard:
    name: str
    domain: str
    built_day: int
    naive_sql: str  # written against the raw feed as it looked on built_day
    mesh_sql: str  # written against the published interface as it looked on built_day
    mesh_major: int


DASHBOARDS = [
    Dashboard(
        "exec_revenue",
        "orders",
        0,
        "select cast(order_ts as date) as d, sum(order_total) from t where status = 'completed' group by 1",
        "select order_date, sum(order_total) from t where status = 'completed' group by 1",
        1,
    ),
    Dashboard(
        "channel_mix",
        "orders",
        25,
        "select sales_channel, count(order_id) from t group by 1",
        "select sales_channel, count(order_id) from t group by 1",
        1,
    ),
    Dashboard(
        "basket_value",
        "orders",
        55,
        "select customer_id, avg(order_amount) from t group by 1",
        "select customer_id, avg(order_amount) from t group by 1",
        2,
    ),
    Dashboard(
        "payments_recon",
        "payments",
        0,
        "select order_id, sum(amount) from t where status = 'captured' group by 1",
        "select order_id, sum(amount) from t where status = 'captured' group by 1",
        1,
    ),
    Dashboard(
        "carrier_scorecard",
        "logistics",
        0,
        "select carrier, avg(case when cast(delivered_at as date) <= promised_date then 1 else 0 end) from t group by 1",
        "select carrier, avg(case when is_on_time then 1 else 0 end) from t group by 1",
        1,
    ),
]


def referenced_columns(sql: str) -> set[str]:
    return {c.name for c in sqlglot.parse_one(sql, dialect="duckdb").find_all(exp.Column)}


def contract_on(domain: str, day: int) -> dict[str, Any]:
    when = START + timedelta(days=day)
    active = [c for c in C.history(domain) if C.created(c) <= when]
    return active[-1]


def mesh_columns(domain: str, major: int, day: int) -> set[str] | None:
    """Columns of the published interface for ``major`` on ``day``; None if that major is not published."""
    latest = contract_on(domain, day)
    if C.SemVer.parse(latest["version"]).major == major:
        return set(C.columns(latest))
    for dp in C.dual_publish(latest):
        if C.SemVer.parse(dp["version"]).major == major:
            return set(C.columns(C.released_version(domain, dp["version"])))
    return None


def replay(n_days: int = N_DAYS) -> dict[str, Any]:
    rows = []
    for db in DASHBOARDS:
        naive_refs, mesh_refs = referenced_columns(db.naive_sql), referenced_columns(db.mesh_sql)
        naive_broken: list[int] = []
        mesh_broken: list[int] = []
        for d in range(db.built_day, n_days):
            raw_cols = set(raw_release_on(db.domain, d).columns)
            if not naive_refs <= raw_cols:
                naive_broken.append(d)
            mcols = mesh_columns(db.domain, db.mesh_major, d)
            if mcols is None or not mesh_refs <= mcols:
                mesh_broken.append(d)
        first_naive = naive_broken[0] if naive_broken else None
        rows.append(
            {
                "dashboard": db.name,
                "domain": db.domain,
                "built_day": db.built_day,
                "live_days": n_days - db.built_day,
                "naive_broken_days": len(naive_broken),
                "naive_first_break": None if first_naive is None else str((START + timedelta(days=first_naive)).date()),
                "naive_missing_columns": sorted(naive_refs - set(raw_release_on(db.domain, n_days - 1).columns)),
                "mesh_interface": f"{C.PRODUCTS[db.domain]}_v{db.mesh_major}",
                "mesh_broken_days": len(mesh_broken),
            }
        )
    live = sum(r["live_days"] for r in rows)
    return {
        "dashboards": rows,
        "dashboard_days": live,
        "naive_broken_dashboard_days": sum(r["naive_broken_days"] for r in rows),
        "mesh_broken_dashboard_days": sum(r["mesh_broken_days"] for r in rows),
        "naive_dashboards_broken": sum(r["naive_broken_days"] > 0 for r in rows),
        "mesh_dashboards_broken": sum(r["mesh_broken_days"] > 0 for r in rows),
    }


def raw_change_summary() -> list[dict[str, Any]]:
    """Every raw-feed schema release after the first, with removed/added columns."""
    from mesh.generators import RAW_RELEASES

    out = []
    for domain, rels in RAW_RELEASES.items():
        for prev, cur in zip(rels, rels[1:], strict=False):
            removed = sorted(set(prev.columns) - set(cur.columns))
            added = sorted(set(cur.columns) - set(prev.columns))
            out.append(
                {
                    "domain": domain,
                    "date": str((START + timedelta(days=cur.day)).date()),
                    "raw_release": cur.tag,
                    "removed": removed,
                    "added": added,
                    "breaking_for_raw_readers": bool(removed),
                    "note": cur.note,
                }
            )
    return out
