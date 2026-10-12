from pathlib import Path

import duckdb

from retail_lakehouse.bronze import load_bronze
from retail_lakehouse.config import SAMPLE_DIR


def test_bronze_load_is_batch_aware_and_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "b.duckdb"
    first = load_bronze(db, SAMPLE_DIR, max_batch="2025-03")
    again = load_bronze(db, SAMPLE_DIR, max_batch="2025-03")
    rest = load_bronze(db, SAMPLE_DIR, max_batch=None)
    assert first["added"]["pos_order_lines"]["rows"] > 0
    assert again["added"]["pos_order_lines"]["rows"] == 0  # nothing loaded twice
    assert rest["added"]["pos_order_lines"]["rows"] > 0
    con = duckdb.connect(str(db), read_only=True)
    try:
        n_files = len(list((SAMPLE_DIR / "pos_order_lines").glob("*.parquet")))
        logged = con.execute(
            "select count(distinct source_file) from bronze._load_log where feed = 'pos_order_lines'"
        ).fetchone()[0]
        assert logged == n_files
        assert con.execute("select max(_load_id) from bronze.pos_order_lines").fetchone()[0] == 3
    finally:
        con.close()
