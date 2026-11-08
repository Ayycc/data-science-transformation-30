import pytest

from semsql.semantic import (
    QuerySpec,
    SemanticError,
    UndefinedDimensionError,
    UndefinedMetricError,
    UnsupportedCombinationError,
)


def test_layer_catalogue(layer):
    assert len(layer.metrics) == 10
    assert layer.is_snapshot("arr_usd") and not layer.is_snapshot("revenue_usd")
    assert "priority" in layer.dimensions_for("support_tickets")
    assert "priority" not in layer.dimensions_for("revenue_usd")
    assert "fiscal_quarter" not in layer.dimensions_for("arr_usd")
    assert "plan" in layer.dimensions_for("revenue_usd")


def test_undefined_metric_and_dimension(layer):
    with pytest.raises(UndefinedMetricError):
        layer.compile(QuerySpec(["nps"]))
    with pytest.raises(UndefinedDimensionError):
        layer.compile(QuerySpec(["revenue_usd"], ["sales_rep"]))
    with pytest.raises(UndefinedMetricError):
        layer.compile(QuerySpec([]))


def test_unsupported_combinations(layer):
    with pytest.raises(UnsupportedCombinationError):
        layer.compile(QuerySpec(["revenue_usd"], ["priority"]))
    with pytest.raises(UnsupportedCombinationError):
        layer.compile(QuerySpec(["arr_usd"], ["fiscal_quarter"]))
    with pytest.raises(UnsupportedCombinationError):
        layer.compile(QuerySpec(["revenue_usd"], as_of="2025-01-31"))
    with pytest.raises(SemanticError):
        layer.compile(QuerySpec(["revenue_usd"], order_by=[["region", "asc"]]))


def test_filter_values_are_escaped(layer, guard):
    sql = layer.compile(QuerySpec(["revenue_usd"], filters={"region": ["EMEA'; DROP TABLE accounts; --"]}))
    assert guard.check(sql).ok  # still a single read-only SELECT


def test_compiled_sql_matches_hand_written_gold(layer, toolbox, con):
    spec = QuerySpec(["revenue_usd", "invoice_count"], ["region"], {"fiscal_year": [2025]})
    res = toolbox.run_query(sql=layer.compile(spec))
    gold = con.execute(
        """SELECT a.region, SUM(i.amount_local * f.rate_to_usd), COUNT(*) FROM invoices i
        JOIN accounts a ON a.account_id = i.account_id
        JOIN fx_rates f ON f.currency = i.currency AND f.month = CAST(date_trunc('month', i.invoice_date) AS DATE)
        WHERE i.status <> 'void' AND NOT a.is_test AND year(i.invoice_date + INTERVAL 11 MONTH) = 2025
        GROUP BY 1 ORDER BY 1"""
    ).fetchall()
    assert len(res["rows"]) == len(gold)
    for (r1, v1, n1), (r2, v2, n2) in zip(res["rows"], gold):
        assert r1 == r2 and n1 == n2 and abs(v1 - v2) < 0.01


def test_multi_model_query_and_ordering(layer, toolbox):
    spec = QuerySpec(
        ["new_accounts", "churned_accounts"],
        ["fiscal_year"],
        order_by=[["new_accounts", "desc"]],
        limit=2,
    )
    res = toolbox.run_query(sql=layer.compile(spec))
    assert res["ok"] and res["columns"] == [
        "fiscal_year",
        "new_accounts",
        "churned_accounts",
    ]
    assert len(res["rows"]) == 2 and res["rows"][0][1] >= res["rows"][1][1]


def test_dim_date_matches_fiscal_arithmetic(con):
    bad = con.execute(
        "SELECT COUNT(*) FROM dim_date WHERE fiscal_year <> year(date_day + INTERVAL 11 MONTH) "
        "OR fiscal_quarter_num <> ((month(date_day) + 10) % 12 // 3 + 1)"
    ).fetchone()[0]
    assert bad == 0
    assert con.execute("SELECT fiscal_quarter FROM dim_date WHERE date_day = DATE '2024-02-01'").fetchone()[0] == "FY2025-Q1"
