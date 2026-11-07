"""Arrow schemas for the source extracts and pandas <-> Arrow conversion."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa

from .generator import Extracts

SCHEMAS = {
    "policy_versions": pa.schema(
        [
            ("load_seq", pa.int64()),
            ("policy_id", pa.int64()),
            ("version_no", pa.int64()),
            ("version_eff", pa.int64()),
            ("lob_code_raw", pa.string()),
            ("region", pa.string()),
            ("load_date", pa.int64()),
        ]
    ),  # fmt: skip
    "premium_txns": pa.schema(
        [
            ("txn_id", pa.int64()),
            ("policy_id", pa.int64()),
            ("txn_type", pa.string()),
            ("txn_eff", pa.int64()),
            ("txn_exp", pa.int64()),
            ("premium_cents", pa.int64()),
            ("booked_date", pa.int64()),
        ]
    ),  # fmt: skip
    "claim_valuations": pa.schema(
        [
            ("claim_id", pa.int64()),
            ("policy_id", pa.int64()),
            ("loss_date", pa.int64()),
            ("report_date", pa.int64()),
            ("valuation_q", pa.int64()),
            ("cum_paid_cents", pa.int64()),
            ("case_reserve_cents", pa.int64()),
        ]
    ),  # fmt: skip
}


def to_arrow(name: str, df: pd.DataFrame) -> pa.Table:
    schema = SCHEMAS[name]
    cols = {}
    for f in schema:
        s = df[f.name]
        if pa.types.is_string(f.type):
            cols[f.name] = pa.array(s.astype(object).tolist(), f.type)
        else:
            cols[f.name] = pa.array([None if pd.isna(v) else int(v) for v in s.tolist()], f.type)
    return pa.table(cols, schema=schema)


def extracts_to_arrow(ex: Extracts) -> dict[str, pa.Table]:
    return {k: to_arrow(k, v) for k, v in ex.as_dict().items()}


def arrow_to_extracts(tables: dict[str, pa.Table]) -> Extracts:
    d = {}
    for k, t in tables.items():
        df = t.to_pandas(types_mapper={pa.int64(): pd.Int64Dtype()}.get)
        for c in df.columns:
            if str(df[c].dtype) == "Int64" and not df[c].isna().any():
                df[c] = df[c].astype("int64")
        d[k] = df
    d["claim_valuations"]["case_reserve_cents"] = d["claim_valuations"]["case_reserve_cents"].astype("Int64")
    return Extracts(d["policy_versions"], d["premium_txns"], d["claim_valuations"])
