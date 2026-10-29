"""Build all four dbt projects on the committed SAMPLE, then check SLOs, lineage and contract enforcement."""

import shutil

import pytest

from mesh import contracts as C
from mesh import graph as G
from mesh import lineage as L
from mesh import slo
from mesh import warehouse as W
from mesh.config import DBT, DBT_PROJECTS, DOMAINS, SAMPLE


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mesh")
    wh = tmp / "wh"
    W.load_raw(raw_dir=SAMPLE, warehouse=wh)
    manifests = {}
    for p in DBT_PROJECTS:
        r = W.dbt_build(p, warehouse=wh, target_path=tmp / f"target_{p}")
        assert r["success"], r
        manifests[p] = W.manifest(p, tmp / f"target_{p}")
    return tmp, wh, manifests


def test_each_product_reports_slo_status(built):
    _, wh, _ = built
    for d in DOMAINS:
        r = slo.evaluate(C.current(d), warehouse=wh)
        assert r["status"] in {"PASS", "FAIL"}
        assert {c["sla"] for c in r["checks"]} == {"freshness", "completeness", "uniqueness", "volume"}
        assert next(c for c in r["checks"] if c["sla"] == "uniqueness")["status"] == "PASS"


def test_gold_metric_traces_back_to_source_columns(built, tmp_path):
    _, wh, manifests = built
    reports = [slo.evaluate(C.current(d), warehouse=wh) for d in DOMAINS]
    events = L.build_events(manifests, W.catalog_schema(wh), reports)
    path = tmp_path / "ev.jsonl"
    L.write_events(events, path)
    ev = L.read_events(path)
    assert all(e["eventType"] in {"START", "COMPLETE"} and e["run"]["runId"] for e in ev)
    t = G.trace(G.build(ev), "analytics.main.mart_daily_kpis", "net_revenue")
    assert set(t["source_columns"]) == {
        "payments/payments_*.csv::amount",
        "payments/payments_*.csv::amount_cents",
        "payments/payments_*.csv::status",
    }
    t_all = G.trace(G.build(ev, include_indirect=True), "analytics.main.mart_daily_kpis", "net_revenue")
    assert "orders/orders_*.csv::order_ts" in t_all["source_columns"]


def test_dbt_contract_blocks_unannounced_rename(built):
    tmp, wh, _ = built
    proj = tmp / "orders_broken"
    shutil.copytree(DBT / "orders", proj, ignore=shutil.ignore_patterns("target", "logs"))
    sql = proj / "models" / "marts" / "fct_orders_v2.sql"
    sql.write_text(sql.read_text().replace("    order_amount,\n", "    order_amount as amount,\n"))
    r = W.dbt_build("orders", warehouse=wh, select="fct_orders", project_dir=proj, target_path=tmp / "t_broken")
    assert not r["success"]
    assert any("contract" in x["message"].lower() for x in r["results"])
