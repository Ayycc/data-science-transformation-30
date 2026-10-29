"""Evaluate each data product's contract SLAs against its published table."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from mesh import contracts as C
from mesh.config import AS_OF, WAREHOUSE
from mesh.warehouse import connect_ro


def evaluate(contract: dict[str, Any], warehouse: Path = WAREHOUSE, as_of: datetime = AS_OF) -> dict[str, Any]:
    domain = contract["domain"]
    table = C.obj(contract)["physicalName"]
    cols = C.columns(contract)
    con = connect_ro(warehouse / f"{domain}.duckdb")
    q = lambda sql: con.execute(sql).fetchone()  # noqa: E731
    n_rows = int(q(f"select count(*) from main.{table}")[0])
    checks: list[dict[str, Any]] = []
    for prop, s in C.sla(contract).items():
        target = s["value"]
        if prop == "freshness":
            col = s["element"].split(".")[-1]
            latest = q(f"select max({col}) from main.{table}")[0]
            observed = round((as_of - latest).total_seconds() / 3600.0, 2)
            ok = observed <= target
            detail = f"latest {col} = {latest}, evaluated at {as_of}"
        elif prop == "completeness":
            req = [c for c, p in cols.items() if p["required"]]
            fracs = {c: q(f"select avg(case when {c} is null then 0.0 else 1.0 end) from main.{table}")[0] for c in req}
            worst = min(fracs, key=fracs.get)
            observed = round(100.0 * float(fracs[worst]), 3)
            ok = observed >= target
            detail = f"min non-null rate over {len(req)} required columns (worst: {worst})"
        elif prop == "uniqueness":
            col = s["element"].split(".")[-1]
            distinct = int(q(f"select count(distinct {col}) from main.{table}")[0])
            observed = round(100.0 * distinct / max(n_rows, 1), 3)
            ok = observed >= target
            detail = f"{distinct} distinct {col} / {n_rows} rows"
        elif prop == "volume":
            observed = n_rows
            ok = observed >= target
            detail = "row count of published table"
        else:  # pragma: no cover - schema forbids other properties
            continue
        checks.append(
            {"sla": prop, "target": target, "unit": s["unit"], "observed": observed, "status": "PASS" if ok else "FAIL", "detail": detail}
        )
    con.close()
    passed = sum(c["status"] == "PASS" for c in checks)
    return {
        "domain": domain,
        "data_product": table,
        "contract_version": contract["version"],
        "owner": C.owner(contract),
        "rows": n_rows,
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "status": "PASS" if passed == len(checks) else "FAIL",
    }
