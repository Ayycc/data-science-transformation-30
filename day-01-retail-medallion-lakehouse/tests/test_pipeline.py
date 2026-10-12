"""Integration tests: the dbt build on the committed sample, and the acceptance criteria."""

from __future__ import annotations


def test_dbt_build_passes_on_fresh_db_and_incrementally(built: dict) -> None:
    for phase in ("phase1_build", "phase2_build"):
        b = built["summary"][phase]
        assert b["success"], b
        assert b["failures"] == []
        assert b["counts"].get("test:pass", 0) >= 60
        assert b["counts"].get("model:success", 0) >= 20


def test_revenue_reconciles_within_tolerance(con) -> None:
    pct, tol, gold, expected = con.execute(
        "select abs_difference_pct, tolerance_pct, gold_revenue, expected_gold_revenue "
        "from audit.audit_revenue_reconciliation"
    ).fetchone()
    assert float(tol) == 0.01
    assert pct <= tol
    assert gold == expected


def test_reconciliation_buckets_match_generator(con, built: dict) -> None:
    import json

    m = json.loads((built["landing"] / "manifest.json").read_text())
    dup, training, unmapped = con.execute(
        "select excluded_duplicate_rows, excluded_training_rows, excluded_unmapped_rows "
        "from audit.audit_revenue_reconciliation"
    ).fetchone()
    assert dup == m["duplicate_rows_injected"]
    assert training == m["training_lines_injected"]
    # unmapped lines that also sit on the training lane count as training
    assert unmapped <= m["unmapped_sku_lines_injected"]


def test_scd2_history_for_renamed_sku(con) -> None:
    recoded = con.execute(
        "select product_id from gold.dim_product group by 1 having count(distinct sku_code) > 1"
    ).fetchall()
    assert recoded, "sample must contain re-coded SKUs"
    for (pid,) in recoded:
        versions = con.execute(
            "select sku_code, valid_from, valid_to, is_current from gold.dim_product "
            "where product_id = ? order by valid_from",
            [pid],
        ).fetchall()
        assert sum(v[3] for v in versions) == 1
        for a, b in zip(versions, versions[1:]):
            assert a[2] == b[1]  # contiguous windows
        # every line sold under the old code is attached to a version carrying that code
        bad = con.execute(
            "select count(*) from gold.fct_order_lines f join gold.dim_product p using (product_sk) "
            "where f.product_id = ? and f.sku_code_at_sale <> p.sku_code",
            [pid],
        ).fetchone()[0]
        assert bad == 0
    assert (
        con.execute(
            "select count(*) from gold.fct_order_lines f join gold.dim_product p using (product_sk) "
            "where p.version_number = 1 and f.product_id in (select product_id from gold.dim_product "
            "group by 1 having count(distinct sku_code) > 1)"
        ).fetchone()[0]
        > 0
    ), "old-code sales should exist before the re-code"


def test_timezone_normalisation_matches_ground_truth(con, built: dict) -> None:
    truth = (built["landing"] / "_truth" / "order_line_truth.parquet").as_posix()
    total, ok, drifted = con.execute(
        f"select count(*), count(*) filter (where f.order_ts_utc = t.true_order_ts_utc), "
        f"count(*) filter (where f.was_exported_in_utc) "
        f"from gold.fct_order_lines f join read_parquet('{truth}') t using (order_line_id)"
    ).fetchone()
    assert drifted > 0
    assert ok == total


def test_late_arrivals_merged_by_incremental_phase(built: dict) -> None:
    p1, p2 = built["summary"]["phase1_stats"], built["summary"]["phase2_stats"]
    assert p2["lines_before_cutoff"] > p1["lines_before_cutoff"]  # late lines for "closed" months
    assert p2["max_load_id"] == 2
    assert p2["returns_awaiting_order_line"] == 0


def test_dq_scorecard_all_pass(con) -> None:
    failed = con.execute("select check_name from audit.dq_scorecard where not passed").fetchall()
    assert failed == []


def test_no_duplicates_in_gold(con) -> None:
    n, d = con.execute("select count(*), count(distinct order_line_id) from gold.fct_order_lines").fetchone()
    assert n == d
