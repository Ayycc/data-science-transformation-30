"""Pandera contracts for the raw feed and the cleaned table."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from .config import PHYSICAL_BOUNDS

raw_schema = pa.DataFrameSchema(
    {
        "station_id": pa.Column(str, pa.Check.str_matches(r"^ST\d{3}$"), nullable=False),
        "ts": pa.Column("datetime64[ns]", nullable=False),
        "pm25": pa.Column(float, nullable=True),
        "no2": pa.Column(float, nullable=True),
        "o3": pa.Column(float, nullable=True),
        "landed_at": pa.Column("datetime64[ns]", nullable=False),
    },
    checks=[pa.Check(lambda df: df["landed_at"] >= df["ts"], error="landed_at before measurement time")],
    strict=True,
    coerce=False,
)


def _range_col(p: str) -> pa.Column:
    lo, hi = PHYSICAL_BOUNDS[p]
    return pa.Column(float, pa.Check.in_range(lo, hi), nullable=True)


cleaned_schema = pa.DataFrameSchema(
    {
        "station_id": pa.Column(str, nullable=False),
        "ts": pa.Column("datetime64[ns]", nullable=False),
        "pm25": _range_col("pm25"),
        "no2": _range_col("no2"),
        "o3": _range_col("o3"),
    },
    unique=["station_id", "ts"],
    strict=False,
)


def validate(schema: pa.DataFrameSchema, df: pd.DataFrame) -> tuple[bool, int, list[str]]:
    """Lazy validation: (passed, number of failure cases, a few example messages)."""
    try:
        schema.validate(df, lazy=True)
        return True, 0, []
    except SchemaErrors as exc:
        fc = exc.failure_cases
        examples = (fc["column"].astype(str) + ": " + fc["check"].astype(str)).drop_duplicates().head(5).tolist()
        return False, int(len(fc)), examples
