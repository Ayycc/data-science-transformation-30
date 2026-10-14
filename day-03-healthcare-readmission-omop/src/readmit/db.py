"""DuckDB loading of the OMOP-like tables and execution of the SQL cohort/feature definitions."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd

from .config import FOLLOWUP_DAYS, LOOKBACK_DAYS, SQL_DIR, WINDOW_END, WINDOW_START
from .normalise import TABLES


def connect(path: Path | str = ":memory:") -> duckdb.DuckDBPyConnection:
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).unlink(missing_ok=True)
    return duckdb.connect(str(path))


def load_omop(con: duckdb.DuckDBPyConnection, frames: dict[str, pd.DataFrame]) -> None:
    for name in TABLES:
        df = frames[name]  # noqa: F841 - referenced by name in the SQL below
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")


def cohort_params() -> dict:
    return {
        "lookback_start": WINDOW_START + timedelta(days=LOOKBACK_DAYS),
        "followup_end": WINDOW_END - timedelta(days=FOLLOWUP_DAYS),
        "horizon": FOLLOWUP_DAYS,
    }


def run_sql(con: duckdb.DuckDBPyConnection, name: str, params: dict | None = None) -> None:
    sql = (SQL_DIR / f"{name}.sql").read_text()
    used = {k: v for k, v in (params or {}).items() if f"${k}" in sql}
    con.execute(sql, used or None)


def build_cohort(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run cohort, feature and LACE SQL. Returns (attrition table, feature table with LACE)."""
    run_sql(con, "cohort", cohort_params())
    run_sql(con, "features")
    run_sql(con, "lace")
    attrition = con.execute(
        """
        SELECT coalesce(exclusion_reason, '99 eligible index admission') AS step, count(*) AS n,
               sum(readmit_30d::INT) AS n_unplanned_readmit_30d
        FROM index_candidates GROUP BY 1 ORDER BY 1
        """
    ).df()
    feats = con.execute(
        """
        SELECT f.*, l.lace_l, l.lace_a, l.lace_c, l.lace_e, l.lace_l + l.lace_a + l.lace_c + l.lace_e AS lace
        FROM features f JOIN lace l USING (visit_occurrence_id) ORDER BY visit_occurrence_id
        """
    ).df()
    return attrition, feats
