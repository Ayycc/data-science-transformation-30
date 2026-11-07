"""Local Apache Iceberg lakehouse via pyiceberg: SQLite catalog + file-system warehouse.

No catalog server, Spark or Docker. Each quarterly close appends the newly booked
source rows to the ``raw`` tables and overwrites the ``reporting`` tables, and tags
every table's snapshot ``close-<quarter>`` so the close can be time-travelled to.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.table import Table
from pyiceberg.types import LongType

from .io import SCHEMAS

# overwrite() on an empty table issues a no-op delete first; that is expected here.
warnings.filterwarnings("ignore", message="Delete operation did not match any records")

REPORT_V1 = pa.schema(
    [("quarter", pa.string()), ("lob", pa.string()), ("region", pa.string()), ("written_premium_cents", pa.int64()),
     ("earned_premium_cents", pa.int64()), ("claims_paid_cents", pa.int64()), ("case_reserve_cents", pa.int64()),
     ("incurred_cents", pa.int64()), ("loss_ratio_bp", pa.int64()), ("policies_in_force", pa.int64())]
)  # fmt: skip
DETAIL_SCHEMAS = {
    "premium_detail": pa.schema(
        [
            ("txn_id", pa.int64()),
            ("quarter", pa.int64()),
            ("policy_id", pa.int64()),
            ("lob", pa.string()),
            ("region", pa.string()),
            ("written_cents", pa.int64()),
            ("earned_ucents", pa.int64()),
        ]
    ),  # fmt: skip
    "claims_detail": pa.schema(
        [
            ("claim_id", pa.int64()),
            ("quarter", pa.int64()),
            ("policy_id", pa.int64()),
            ("lob", pa.string()),
            ("region", pa.string()),
            ("paid_cents", pa.int64()),
            ("case_reserve_cents", pa.int64()),
            ("incurred_cum_cents", pa.int64()),
            ("incurred_cents", pa.int64()),
        ]
    ),  # fmt: skip
}


def tag_for(label: str) -> str:
    return f"close-{label}"


def df_to_arrow(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    cols = {}
    for f in schema:
        vals = df[f.name].tolist()
        if pa.types.is_string(f.type):
            cols[f.name] = pa.array([None if v is None or v is pd.NA else str(v) for v in vals], f.type)
        else:
            cols[f.name] = pa.array([None if pd.isna(v) else int(v) for v in vals], f.type)
    return pa.table(cols, schema=schema)


class Lakehouse:
    def __init__(self, root: Path, fresh: bool = True):
        self.root = Path(root)
        if fresh and self.root.exists():
            shutil.rmtree(self.root)
        (self.root / "warehouse").mkdir(parents=True, exist_ok=True)
        self.catalog = SqlCatalog("local", uri=f"sqlite:///{self.root / 'catalog.db'}", warehouse=(self.root / "warehouse").resolve().as_uri())
        for ns in ("raw", "reporting"):
            if (ns,) not in self.catalog.list_namespaces():
                self.catalog.create_namespace(ns)

    def table(self, ident: str) -> Table:
        return self.catalog.load_table(ident)

    def _ensure(self, ident: str, schema: pa.Schema) -> Table:
        if not self.catalog.table_exists(ident):
            return self.catalog.create_table(ident, schema=schema)
        return self.table(ident)

    def _tag(self, tbl: Table, label: str) -> None:
        tbl = self.table(str(".".join(tbl.name()[-2:])))
        tbl.manage_snapshots().create_tag(tbl.current_snapshot().snapshot_id, tag_for(label)).commit()

    # ------------------------------------------------------------------ raw layer
    def append_raw(self, name: str, increment: pa.Table, label: str) -> None:
        tbl = self._ensure(f"raw.{name}", SCHEMAS[name])
        tbl.append(increment, snapshot_properties={"close": label})
        self._tag(tbl, label)

    def read_raw(self, name: str, label: str | None = None) -> pa.Table:
        tbl = self.table(f"raw.{name}")
        snap = tbl.snapshot_by_name(tag_for(label)).snapshot_id if label else None
        return tbl.scan(snapshot_id=snap).to_arrow()

    # ------------------------------------------------------------------ reporting layer
    def write_report(self, report: pd.DataFrame, label: str, with_ibnr: bool) -> list[str]:
        """Overwrite the quarterly report; evolve the schema (add ibnr_cents) the first time it is required."""
        events = []
        tbl = self._ensure("reporting.quarterly_report", REPORT_V1)
        if with_ibnr and "ibnr_cents" not in [f.name for f in tbl.schema().fields]:
            with tbl.update_schema() as upd:
                upd.add_column("ibnr_cents", LongType(), doc="IBNR estimate, required from 2025Q1")
            tbl = self.table("reporting.quarterly_report")
            events.append(f"schema evolved at close {label}: added ibnr_cents (schema id {tbl.schema().schema_id})")
        schema = REPORT_V1.append(pa.field("ibnr_cents", pa.int64())) if "ibnr_cents" in [f.name for f in tbl.schema().fields] else REPORT_V1
        tbl.overwrite(df_to_arrow(report, schema), snapshot_properties={"close": label})
        self._tag(tbl, label)
        return events

    def write_detail(self, name: str, df: pd.DataFrame, label: str) -> None:
        tbl = self._ensure(f"reporting.{name}", DETAIL_SCHEMAS[name])
        tbl.overwrite(df_to_arrow(df, DETAIL_SCHEMAS[name]), snapshot_properties={"close": label})
        self._tag(tbl, label)

    def read_report(self, label: str | None = None) -> pd.DataFrame:
        tbl = self.table("reporting.quarterly_report")
        snap = tbl.snapshot_by_name(tag_for(label)).snapshot_id if label else None
        return tbl.scan(snapshot_id=snap).to_arrow().to_pandas(types_mapper={pa.int64(): pd.Int64Dtype()}.get)

    def report_schema_at(self, label: str) -> list[str]:
        tbl = self.table("reporting.quarterly_report")
        snap = tbl.snapshot_by_name(tag_for(label))
        return [f.name for f in tbl.schemas()[snap.schema_id].fields]

    def data_bytes(self, namespace: str) -> int:
        base = self.root / "warehouse" / namespace
        return sum(p.stat().st_size for p in base.rglob("*.parquet")) if base.exists() else 0
