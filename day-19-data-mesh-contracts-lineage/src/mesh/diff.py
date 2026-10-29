"""Contract diff and semantic-version gate.

Rules are written from the *consumer's* point of view: a change is breaking if a query or
expectation that was valid against the old contract can fail, or silently change meaning, against
the new one.

=========================================  ========
change                                     level
=========================================  ========
schema object / column removed or renamed  major
column physical type changed (not widened) major
column required -> optional                major
primary key changed                        major
enum value removed                         major
physicalName changed without major bump    major
SLA weakened (freshness up, % down)        major
dual-published version retired early      major
dual-published version retired after date minor
SLA removed                                major
column added (optional or required)        minor
column type widened (int->bigint, wider decimal) minor
column optional -> required                minor
enum value added                           minor
SLA tightened / added                      minor
description, owner, team, classification   patch
=========================================  ========

The gate passes when the new version is greater than the old one *and* the bump is at least the
highest level found. A major release must keep the previous major published (``dualPublish``) for a
deprecation window. Dual-published interfaces are rendered from the immutable files in ``history/``;
``scripts/check_contracts.py`` fails if a released history file is edited.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from mesh import contracts as C
from mesh.config import AS_OF

LEVELS = {"none": 0, "patch": 1, "minor": 2, "major": 3}


@dataclass
class Change:
    level: str
    kind: str
    path: str
    detail: str


@dataclass
class GateResult:
    domain: str
    old_version: str
    new_version: str
    required_bump: str
    actual_bump: str
    passed: bool
    changes: list[Change] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["changes"] = [asdict(c) for c in self.changes]
        return d


_DEC = re.compile(r"decimal\((\d+),(\d+)\)")
_INT_ORDER = ["integer", "bigint"]


def _widened(old: str, new: str) -> bool:
    if old == new:
        return False
    if old in _INT_ORDER and new in _INT_ORDER:
        return _INT_ORDER.index(new) > _INT_ORDER.index(old)
    mo, mn = _DEC.fullmatch(old), _DEC.fullmatch(new)
    if mo and mn:
        po, so = map(int, mo.groups())
        pn, sn = map(int, mn.groups())
        return sn >= so and (pn - sn) >= (po - so) and (pn, sn) != (po, so)
    if old in _INT_ORDER and mn:
        return int(mn.group(1)) - int(mn.group(2)) >= (10 if old == "integer" else 19)
    return False


# For an SLA, which direction is "stronger" for consumers.
_SLA_LOWER_IS_STRONGER = {"freshness"}


def diff(old: dict[str, Any], new: dict[str, Any], today: date | None = None) -> list[Change]:
    today = today or AS_OF.date()
    ch: list[Change] = []
    add = lambda level, kind, path, detail: ch.append(Change(level, kind, path, detail))  # noqa: E731

    old_objs = {o["name"]: o for o in old["schema"]}
    new_objs = {o["name"]: o for o in new["schema"]}
    for name in old_objs.keys() - new_objs.keys():
        add("major", "object_removed", f"schema/{name}", "schema object removed")
    for name in new_objs.keys() - old_objs.keys():
        add("minor", "object_added", f"schema/{name}", "schema object added")

    for name in old_objs.keys() & new_objs.keys():
        o, n = old_objs[name], new_objs[name]
        if o["physicalName"] != n["physicalName"]:
            add("major", "physical_name_changed", f"schema/{name}", f"{o['physicalName']} -> {n['physicalName']}")
        oc = {p["name"]: p for p in o["properties"]}
        nc = {p["name"]: p for p in n["properties"]}
        removed, added = oc.keys() - nc.keys(), nc.keys() - oc.keys()
        for col in sorted(removed):
            hint = [a for a in added if nc[a]["physicalType"] == oc[col]["physicalType"]]
            detail = f"column removed (possible rename to {', '.join(sorted(hint))})" if hint else "column removed"
            add("major", "column_removed", f"schema/{name}/{col}", detail)
        for col in sorted(added):
            add("minor", "column_added", f"schema/{name}/{col}", f"{'required' if nc[col]['required'] else 'optional'} column added")
        o_pk = sorted(c for c, p in oc.items() if p.get("primaryKey"))
        n_pk = sorted(c for c, p in nc.items() if p.get("primaryKey"))
        if o_pk != n_pk:
            add("major", "primary_key_changed", f"schema/{name}", f"{o_pk} -> {n_pk}")
        for col in sorted(oc.keys() & nc.keys()):
            a, b = oc[col], nc[col]
            path = f"schema/{name}/{col}"
            if a["physicalType"] != b["physicalType"]:
                if _widened(a["physicalType"], b["physicalType"]):
                    add("minor", "type_widened", path, f"{a['physicalType']} -> {b['physicalType']}")
                else:
                    add("major", "type_changed", path, f"{a['physicalType']} -> {b['physicalType']}")
            elif a["logicalType"] != b["logicalType"]:
                add("major", "type_changed", path, f"logical {a['logicalType']} -> {b['logicalType']}")
            if a["required"] and not b["required"]:
                add("major", "required_relaxed", path, "required -> optional (nulls may now appear)")
            elif not a["required"] and b["required"]:
                add("minor", "required_tightened", path, "optional -> required")
            if bool(a.get("unique")) and not bool(b.get("unique")) and not b.get("primaryKey"):
                add("major", "unique_relaxed", path, "unique guarantee removed")
            ea = set((a.get("logicalTypeOptions") or {}).get("enum") or [])
            eb = set((b.get("logicalTypeOptions") or {}).get("enum") or [])
            if ea and not eb:
                add("minor", "enum_removed", path, "accepted-values constraint dropped (wider domain)")
            elif ea or eb:
                if ea - eb:
                    add("major", "enum_value_removed", path, f"values removed: {sorted(ea - eb)}")
                if eb - ea and ea:
                    add("minor", "enum_value_added", path, f"values added: {sorted(eb - ea)}")
                if eb and not ea:
                    add("minor", "enum_added", path, "accepted-values constraint added")
            for k in ("description", "classification"):
                if a.get(k) != b.get(k):
                    add("patch", f"{k}_changed", path, f"{k} changed")

    os_, ns = C.sla(old), C.sla(new)
    for prop in os_.keys() - ns.keys():
        add("major", "sla_removed", f"sla/{prop}", "SLA removed")
    for prop in ns.keys() - os_.keys():
        add("minor", "sla_added", f"sla/{prop}", "SLA added")
    for prop in os_.keys() & ns.keys():
        a, b = os_[prop]["value"], ns[prop]["value"]
        if a == b:
            continue
        weaker = b > a if prop in _SLA_LOWER_IS_STRONGER else b < a
        add("major" if weaker else "minor", "sla_weakened" if weaker else "sla_tightened", f"sla/{prop}", f"{a} -> {b}")

    od, nd = {d["version"]: d for d in C.dual_publish(old)}, {d["version"]: d for d in C.dual_publish(new)}
    for v in od.keys() - nd.keys():
        due = date.fromisoformat(od[v]["deprecationDate"]) <= today
        if due:
            add("minor", "dual_publish_ended", f"customProperties/dualPublish/{v}", f"v{v} retired after its deprecation date")
        else:
            add("major", "dual_publish_ended_early", f"customProperties/dualPublish/{v}", f"v{v} retired before {od[v]['deprecationDate']}")
    for v in nd.keys() - od.keys():
        add("minor", "dual_publish_started", f"customProperties/dualPublish/{v}", f"v{v} kept published until {nd[v]['deprecationDate']}")
    for v in od.keys() & nd.keys():
        if od[v]["deprecationDate"] > nd[v]["deprecationDate"]:
            add(
                "major",
                "deprecation_shortened",
                f"customProperties/dualPublish/{v}",
                f"{od[v]['deprecationDate']} -> {nd[v]['deprecationDate']}",
            )
        elif od[v] != nd[v]:
            add("patch", "deprecation_extended", f"customProperties/dualPublish/{v}", "deprecation window changed")

    for k in ("description", "team", "name", "tenant", "servers"):
        if old.get(k) != new.get(k):
            add("major" if k == "servers" else "patch", f"{k}_changed", k, f"{k} changed")
    if old.get("status") != new.get("status"):
        add("minor", "status_changed", "status", f"{old.get('status')} -> {new.get('status')}")
    return ch


def bump_level(old_v: str, new_v: str) -> str:
    a, b = C.SemVer.parse(old_v), C.SemVer.parse(new_v)
    if b.as_tuple() <= a.as_tuple():
        return "none"
    if b.major > a.major:
        return "major"
    if b.minor > a.minor:
        return "minor"
    return "patch"


def gate(old: dict[str, Any], new: dict[str, Any], today: date | None = None) -> GateResult:
    errors = C.validate(new)
    changes = diff(old, new, today)
    required = max((c.level for c in changes), key=LEVELS.__getitem__, default="none")
    actual = bump_level(old["version"], new["version"])
    if actual == "none" and (changes or old["version"] != new["version"]):
        errors.append(f"version must increase: {old['version']} -> {new['version']}")
    if LEVELS[actual] < LEVELS[required]:
        errors.append(f"{required} change(s) need at least a {required} bump; got {actual} ({old['version']} -> {new['version']})")
    if actual == "major":
        kept = {d["physicalName"] for d in C.dual_publish(new)}
        old_iface = C.obj(old)["physicalName"]
        if old_iface != C.obj(new)["physicalName"] and old_iface not in kept:
            errors.append(f"major release must dual-publish {old_iface} (customProperties/dualPublish) for a deprecation window")
    if old["domain"] != new["domain"] or old["id"] != new["id"]:
        errors.append("contract id/domain changed: publish a new data product instead")
    return GateResult(
        domain=new["domain"],
        old_version=old["version"],
        new_version=new["version"],
        required_bump=required,
        actual_bump=actual,
        passed=not errors,
        changes=changes,
        errors=errors,
    )
