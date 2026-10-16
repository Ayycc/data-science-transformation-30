"""Load the raw extract into DuckDB and build the typed staging table (no rows are dropped here)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from . import geo

STAGE_SQL = """
CREATE OR REPLACE TABLE staged AS
SELECT
    CAST(rowid + 1 AS BIGINT)                          AS source_line,
    sr_number,
    sr_type                                            AS sr_type_raw,
    owner_department,
    status,
    origin,
    created_date                                       AS created_raw,
    closed_date                                        AS closed_raw,
    TRY_STRPTIME(created_date, '%Y-%m-%d %H:%M:%S')    AS created_ts,
    TRY_STRPTIME(closed_date, '%Y-%m-%d %H:%M:%S')     AS closed_ts,
    ward                                               AS ward_raw,
    TRY_CAST(ward AS INTEGER)                          AS ward,
    TRY_CAST(latitude AS DOUBLE)                       AS latitude,
    TRY_CAST(longitude AS DOUBLE)                      AS longitude
FROM raw
ORDER BY source_line
"""


def load_raw(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    """Read every field as text, exactly as exported. `source_line` is the 1-based data line in the file."""
    con.execute("SET preserve_insertion_order = true")
    con.execute(
        "CREATE OR REPLACE TABLE raw AS SELECT * FROM read_csv(?, header = true, all_varchar = true, "
        "quote = '\"', escape = '\"', parallel = false)",
        [str(path)],
    )
    con.execute(STAGE_SQL)
    return con.execute("SELECT count(*) FROM staged").fetchone()[0]


def staged_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("SELECT * FROM staged ORDER BY source_line").df()
    df["ward"] = df["ward"].astype("Int64")
    for c in ("created_ts", "closed_ts"):
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
    # spatial helper flags (shapely, vectorised) used by both validation layers
    lon, lat = df.longitude.to_numpy(float), df.latitude.to_numpy(float)
    df["in_bbox"] = geo.in_bbox(lon, lat)
    df["in_city_polygon"] = geo.in_city(lon, lat)
    df["swapped_in_bbox"] = geo.in_bbox(lat, lon)
    return df
