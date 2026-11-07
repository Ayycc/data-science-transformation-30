from legacy28.config import QUARTERS
from legacy28.generator import generate
from legacy28.iceberg_store import Lakehouse
from legacy28.pipeline import parallel_run, time_travel_checks
from legacy28.reconcile import diff_cells


def test_parallel_run_and_time_travel(tmp_path):
    ex = generate(600, seed=11)
    lake = Lakehouse(tmp_path / "lake")
    pr = parallel_run(ex, lake, log=lambda *_: None)
    assert all(c["compat_equals_legacy"] for c in pr["closes"])
    assert all(c["duckdb_equals_polars"] for c in pr["closes"])
    assert sum(c["unexplained_differences"] for c in pr["closes"]) == 0
    assert all(v["untraced"] == 0 for v in pr["trace"].values())

    tt = time_travel_checks(lake, pr, log=lambda *_: None)
    n = len(QUARTERS)
    assert tt["filed_quarter_reproduced"] == n
    assert tt["rerun_from_time_travelled_inputs_reproduces_report"] == n
    assert tt["schema_version_correct_at_each_close"] == n
    assert tt["rollback_drill_ok"]

    # previous quarter's filed return, straight from the tag
    prev = QUARTERS[-2].label
    snap = lake.read_report(prev)
    filed = (
        snap[snap.quarter == prev].astype({"quarter": object, "lob": object, "region": object}).sort_values(["quarter", "lob", "region"]).reset_index(drop=True)
    )
    assert diff_cells(filed, pr["filed_modern"][prev], ["quarter", "lob", "region"]).empty

    # schema evolution: the 2024Q4 snapshot has no IBNR column, 2025Q1 onwards does
    assert "ibnr_cents" not in lake.read_report("2024Q4").columns
    assert "ibnr_cents" in lake.read_report("2025Q1").columns
    assert lake.read_report("2025Q1")["ibnr_cents"].notna().any()
    assert len(lake.table("reporting.quarterly_report").schemas()) == 2


def test_raw_tables_append_per_close(tmp_path):
    ex = generate(300, seed=3)
    lake = Lakehouse(tmp_path / "lake")
    from legacy28.generator import as_of_close
    from legacy28.io import to_arrow

    prev = 0
    for q in QUARTERS[:3]:
        df = as_of_close(ex, q.idx).premium_txns
        inc = df[df["txn_id"].isin(set(df["txn_id"]) - set(as_of_close(ex, q.idx - 1).premium_txns["txn_id"]) if q.idx > 1 else set(df["txn_id"]))]
        lake.append_raw("premium_txns", to_arrow("premium_txns", inc), q.label)
        assert lake.read_raw("premium_txns").num_rows == len(df) >= prev
        prev = len(df)
    assert lake.read_raw("premium_txns", QUARTERS[0].label).num_rows == len(as_of_close(ex, 1).premium_txns)
