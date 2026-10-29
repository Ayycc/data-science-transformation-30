"""Load and validate ODCS-style contracts, and render them into dbt model-contract YAML.

The contract YAML in ``contracts/`` is the single source of truth. dbt's model ``_contract.yml``
files are *generated* from it (``render_dbt_yaml``) so the producer cannot drift from what consumers
were promised: dbt's enforced model contract then fails the build if the SQL output differs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from mesh.config import CONTRACTS, PRODUCTS


@dataclass(frozen=True)
class SemVer:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, s: str) -> SemVer:
        a, b, c = (int(x) for x in s.split("."))
        return cls(a, b, c)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)


@cache
def _schema() -> dict[str, Any]:
    return json.loads((CONTRACTS / "odcs-lite.schema.json").read_text())


def validate(contract: dict[str, Any]) -> list[str]:
    """JSON-Schema validation plus semantic checks. Returns a list of error strings (empty = valid)."""
    v = jsonschema.Draft202012Validator(_schema(), format_checker=jsonschema.FormatChecker())
    errors = [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in v.iter_errors(contract)]
    if errors:
        return errors
    for obj in contract["schema"]:
        names = [p["name"] for p in obj["properties"]]
        if len(names) != len(set(names)):
            errors.append(f"schema/{obj['name']}: duplicate property names")
        pks = [p["name"] for p in obj["properties"] if p.get("primaryKey")]
        if not pks:
            errors.append(f"schema/{obj['name']}: no primaryKey declared")
        for p in obj["properties"]:
            if p.get("primaryKey") and not p.get("required"):
                errors.append(f"schema/{obj['name']}/{p['name']}: primary key must be required")
        major = SemVer.parse(contract["version"]).major
        if not obj["physicalName"].endswith(f"_v{major}"):
            errors.append(f"schema/{obj['name']}: physicalName {obj['physicalName']} must end with _v{major} (major version)")
    for dp in dual_publish(contract):
        if SemVer.parse(dp["version"]).major >= SemVer.parse(contract["version"]).major:
            errors.append(f"customProperties/dualPublish: {dp['version']} must be an older major than {contract['version']}")
    return errors


def load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def loads(text: str) -> dict[str, Any]:
    return yaml.safe_load(text)


def current_path(domain: str) -> Path:
    return CONTRACTS / domain / f"{PRODUCTS[domain]}.odcs.yaml"


def current(domain: str) -> dict[str, Any]:
    return load(current_path(domain))


def history(domain: str) -> list[dict[str, Any]]:
    """All released versions of a domain's contract, oldest first."""
    files = (CONTRACTS / domain / "history").glob("*.odcs.yaml")
    out = [load(f) for f in files]
    return sorted(out, key=lambda c: SemVer.parse(c["version"]).as_tuple())


def released_version(domain: str, version: str) -> dict[str, Any]:
    for c in history(domain):
        if c["version"] == version:
            return c
    raise KeyError(f"{domain} {version} not in history")


def created(contract: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(contract["contractCreatedTs"].replace("Z", "+00:00")).replace(tzinfo=None)


def dual_publish(contract: dict[str, Any]) -> list[dict[str, Any]]:
    for cp in contract.get("customProperties", []) or []:
        if cp["property"] == "dualPublish":
            return list(cp["value"])
    return []


def obj(contract: dict[str, Any]) -> dict[str, Any]:
    return contract["schema"][0]


def columns(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {p["name"]: p for p in obj(contract)["properties"]}


def sla(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["property"]: s for s in contract["slaProperties"]}


def owner(contract: dict[str, Any]) -> str:
    return next(m["username"] for m in contract["team"]["members"] if m["role"] == "owner")


def published_interfaces(contract: dict[str, Any]) -> list[tuple[str, dict[str, Any], str | None]]:
    """(physicalName, contract-for-that-interface, deprecationDate) for latest + dual-published versions."""
    out: list[tuple[str, dict[str, Any], str | None]] = [(obj(contract)["physicalName"], contract, None)]
    for dp in dual_publish(contract):
        old = released_version(contract["domain"], dp["version"])
        out.append((dp["physicalName"], old, dp["deprecationDate"]))
    return out


# ---------------------------------------------------------------- dbt rendering


def _dbt_columns(contract: dict[str, Any]) -> list[dict[str, Any]]:
    cols = []
    for p in obj(contract)["properties"]:
        c: dict[str, Any] = {"name": p["name"], "data_type": p["physicalType"], "description": p.get("description", "")}
        if p.get("required"):
            c["constraints"] = [{"type": "not_null"}]
        tests: list[Any] = []
        if p.get("unique") or p.get("primaryKey"):
            tests.append("unique")
        enum = (p.get("logicalTypeOptions") or {}).get("enum")
        if enum:
            tests.append({"accepted_values": {"arguments": {"values": enum}}})
        if tests:
            c["data_tests"] = tests
        cols.append(c)
    return cols


def render_dbt_yaml(contract: dict[str, Any]) -> str:
    """dbt model YAML with an enforced contract and one dbt model version per published major."""
    name = obj(contract)["name"]
    latest_major = SemVer.parse(contract["version"]).major
    versions: list[dict[str, Any]] = [{"v": latest_major, "columns": _dbt_columns(contract)}]
    for dp in dual_publish(contract):
        old = released_version(contract["domain"], dp["version"])
        versions.append(
            {
                "v": SemVer.parse(dp["version"]).major,
                "deprecation_date": dp["deprecationDate"],
                "columns": _dbt_columns(old),
            }
        )
    model = {
        "name": name,
        "description": contract["description"]["purpose"],
        "latest_version": latest_major,
        "access": "public",
        "config": {
            "materialized": "table",
            "contract": {"enforced": True},
            "meta": {"contract_id": contract["id"], "contract_version": contract["version"], "owner": owner(contract)},
        },
        "versions": versions,
    }
    header = f"# GENERATED from contracts/{contract['domain']}/{PRODUCTS[contract['domain']]}.odcs.yaml - do not edit by hand.\n"
    return header + yaml.safe_dump({"version": 2, "models": [model]}, sort_keys=False, width=120)


def dbt_yaml_path(domain: str) -> Path:
    from mesh.config import DBT

    return DBT / domain / "models" / "marts" / "_contract.yml"
