"""A labelled suite of proposed contract changes used to exercise the CI gate.

Each scenario mutates a copy of a *current* contract the way a domain engineer might in a pull
request, and states whether a correct gate should accept it. The suite is hand-written (not
sampled), so its pass rate measures rule coverage, not real-world precision.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mesh import contracts as C
from mesh import diff


@dataclass(frozen=True)
class Scenario:
    name: str
    domain: str
    new_version: str
    mutate: Callable[[dict[str, Any]], None]
    should_pass: bool
    why: str


def _col(c: dict[str, Any], name: str) -> dict[str, Any]:
    return C.columns(c)[name]


def _rename(c: dict[str, Any], old: str, new: str) -> None:
    _col(c, old)["name"] = new


def _drop(c: dict[str, Any], name: str) -> None:
    C.obj(c)["properties"] = [p for p in C.obj(c)["properties"] if p["name"] != name]


def _add(c: dict[str, Any], name: str, required: bool = False) -> None:
    C.obj(c)["properties"].append(
        {"name": name, "logicalType": "string", "physicalType": "varchar", "required": required, "description": "new"}
    )


def _sla(c: dict[str, Any], prop: str, value: float) -> None:
    C.sla(c)[prop]["value"] = value


def _phys(c: dict[str, Any], major: int) -> None:
    obj = C.obj(c)
    obj["physicalName"] = obj["physicalName"].rsplit("_v", 1)[0] + f"_v{major}"


def _major_with_dual(c: dict[str, Any], major: int) -> None:
    old_phys, old_ver = C.obj(c)["physicalName"], c["version"]
    _phys(c, major)
    c["customProperties"] = [
        {"property": "dualPublish", "value": [{"version": old_ver, "physicalName": old_phys, "deprecationDate": "2027-01-31"}]}
    ]


SCENARIOS: list[Scenario] = [
    Scenario("rename_amount_minor", "payments", "1.3.0", lambda c: _rename(c, "amount", "amount_usd"), False, "column rename is breaking"),
    Scenario("rename_amount_patch", "payments", "1.2.1", lambda c: _rename(c, "amount", "amount_usd"), False, "column rename is breaking"),
    Scenario(
        "rename_amount_major_dual",
        "payments",
        "2.0.0",
        lambda c: (_rename(c, "amount", "amount_usd"), _major_with_dual(c, 2)),
        True,
        "major bump + v1 kept published",
    ),
    Scenario(
        "rename_amount_major_no_dual",
        "payments",
        "2.0.0",
        lambda c: (_rename(c, "amount", "amount_usd"), _phys(c, 2)),
        False,
        "major release must dual-publish the previous major",
    ),
    Scenario("drop_sales_channel_patch", "orders", "2.1.1", lambda c: _drop(c, "sales_channel"), False, "column removal is breaking"),
    Scenario(
        "narrow_type_minor",
        "orders",
        "2.2.0",
        lambda c: _col(c, "order_amount").update(physicalType="integer", logicalType="integer"),
        False,
        "decimal -> integer loses precision",
    ),
    Scenario(
        "widen_decimal_minor",
        "orders",
        "2.2.0",
        lambda c: _col(c, "order_amount").update(physicalType="decimal(18,2)"),
        True,
        "widening is additive",
    ),
    Scenario(
        "widen_decimal_patch",
        "orders",
        "2.1.1",
        lambda c: _col(c, "order_amount").update(physicalType="decimal(18,2)"),
        False,
        "widening needs a minor",
    ),
    Scenario(
        "relax_required_minor",
        "logistics",
        "1.2.0",
        lambda c: _col(c, "delivered_at").update(required=False),
        False,
        "nulls may now appear",
    ),
    Scenario(
        "tighten_required_minor",
        "logistics",
        "1.2.0",
        lambda c: _col(c, "carrier_service").update(required=True),
        True,
        "stronger guarantee",
    ),
    Scenario("loosen_freshness_patch", "logistics", "1.1.1", lambda c: _sla(c, "freshness", 96), False, "weaker SLA is breaking"),
    Scenario("tighten_freshness_minor", "logistics", "1.2.0", lambda c: _sla(c, "freshness", 24), True, "stronger SLA"),
    Scenario(
        "add_enum_value_minor",
        "orders",
        "2.2.0",
        lambda c: _col(c, "status")["logicalTypeOptions"]["enum"].append("on_hold"),
        True,
        "new value is additive",
    ),
    Scenario(
        "remove_enum_value_minor",
        "orders",
        "2.2.0",
        lambda c: _col(c, "status")["logicalTypeOptions"]["enum"].remove("cancelled"),
        False,
        "consumers filtering on it silently get 0 rows",
    ),
    Scenario("add_optional_column_minor", "payments", "1.3.0", lambda c: _add(c, "gateway"), True, "additive"),
    Scenario("add_optional_column_patch", "payments", "1.2.1", lambda c: _add(c, "gateway"), False, "additive needs a minor"),
    Scenario(
        "change_primary_key_major_dual",
        "orders",
        "3.0.0",
        lambda c: (
            _col(c, "order_id").update(primaryKey=False),
            _col(c, "customer_id").update(primaryKey=True, unique=True),
            _major_with_dual(c, 3),
        ),
        True,
        "PK change with major bump + dual publish",
    ),
    Scenario(
        "change_primary_key_minor",
        "orders",
        "2.2.0",
        lambda c: (_col(c, "order_id").update(primaryKey=False), _col(c, "customer_id").update(primaryKey=True, unique=True)),
        False,
        "PK change is breaking",
    ),
    Scenario(
        "description_patch",
        "logistics",
        "1.1.1",
        lambda c: _col(c, "carrier").update(description="Carrier SCAC-like code."),
        True,
        "metadata only",
    ),
    Scenario(
        "owner_handover_patch",
        "payments",
        "1.2.1",
        lambda c: c["team"]["members"].append({"username": "payments-steward", "role": "steward"}),
        True,
        "metadata only",
    ),
    Scenario("no_version_bump", "payments", "1.2.0", lambda c: _add(c, "gateway"), False, "version must increase"),
    Scenario(
        "retire_v1_early",
        "orders",
        "2.2.0",
        lambda c: c.pop("customProperties"),
        False,
        "orders v1 is still promised until 2026-10-31",
    ),
    Scenario(
        "major_without_physical_rename",
        "logistics",
        "2.0.0",
        lambda c: _drop(c, "carrier_service"),
        False,
        "physicalName must carry the major",
    ),
    Scenario(
        "invalid_contract_no_owner",
        "logistics",
        "1.1.1",
        lambda c: c["team"].update(members=[{"username": "x", "role": "engineer"}]),
        False,
        "schema: owner required",
    ),
]


def apply(s: Scenario) -> tuple[dict[str, Any], dict[str, Any]]:
    old = C.current(s.domain)
    new = copy.deepcopy(old)
    s.mutate(new)
    new["version"] = s.new_version
    return old, new


def run_suite() -> list[dict[str, Any]]:
    out = []
    for s in SCENARIOS:
        old, new = apply(s)
        r = diff.gate(old, new)
        out.append(
            {
                "scenario": s.name,
                "domain": s.domain,
                "version": f"{old['version']} -> {s.new_version}",
                "expected": "pass" if s.should_pass else "fail",
                "gate": "pass" if r.passed else "fail",
                "correct": r.passed == s.should_pass,
                "required_bump": r.required_bump,
                "actual_bump": r.actual_bump,
                "why": s.why,
                "first_error": r.errors[0] if r.errors else "",
            }
        )
    return out


def replay_history() -> list[dict[str, Any]]:
    """Gate every real release transition in contracts/*/history (all should pass)."""
    out = []
    for domain in C.PRODUCTS:
        hist = C.history(domain)
        for old, new in zip(hist, hist[1:], strict=False):
            r = diff.gate(old, new)
            out.append(
                {
                    "domain": domain,
                    "version": f"{old['version']} -> {new['version']}",
                    "date": new["contractCreatedTs"][:10],
                    "required_bump": r.required_bump,
                    "actual_bump": r.actual_bump,
                    "passed": r.passed,
                    "changes": [f"{c.level}:{c.kind}:{c.path}" for c in r.changes],
                    "errors": r.errors,
                }
            )
    return out
