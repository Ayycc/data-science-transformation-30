"""End-to-end: data -> raw load -> contracts -> dbt (3 domains + consumer) -> SLOs -> OpenLineage ->
column lineage graph -> CI gate replay + scenarios -> consumer breakage replay -> catalogue -> results/."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import duckdb  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import generate_data  # noqa: E402
from mesh import catalog, consumers, scenarios, slo  # noqa: E402
from mesh import contracts as C  # noqa: E402
from mesh import graph as G  # noqa: E402
from mesh import lineage as L  # noqa: E402
from mesh import warehouse as W  # noqa: E402
from mesh.config import AS_OF, DBT, DBT_PROJECTS, DOMAINS, N_DAYS, RESULTS, SEED, START, WAREHOUSE  # noqa: E402

GOLD = ("analytics.main.mart_daily_kpis", "net_revenue")


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list | dict) else v) for k, v in r.items()})


def dbt_contract_break_demo() -> dict:
    """Rename a column in the orders v2 SQL without touching the contract: dbt must refuse to build."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        proj = tmp / "orders"
        shutil.copytree(DBT / "orders", proj, ignore=shutil.ignore_patterns("target", "logs"))
        sql = proj / "models" / "marts" / "fct_orders_v2.sql"
        sql.write_text(sql.read_text().replace("    order_amount,\n", "    order_amount as amount,\n"))
        wh = tmp / "wh"
        wh.mkdir()
        shutil.copy(WAREHOUSE / "orders.duckdb", wh / "orders.duckdb")
        r = W.dbt_build("orders", warehouse=wh, select="fct_orders", project_dir=proj, target_path=tmp / "target")
    msgs = " | ".join(x["message"] for x in r["results"] if x["status"] == "error")
    return {
        "change": "fct_orders_v2.sql: order_amount -> amount (contract unchanged)",
        "dbt_build_success": r["success"],
        "message": msgs[:600],
    }


def main() -> None:
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)
    (RESULTS / "lineage").mkdir(exist_ok=True)

    print("[1/9] generate SYNTHETIC domain feeds")
    data_info = generate_data.main()

    print("[2/9] land raw feeds in per-domain DuckDB files")
    raw = W.load_raw()

    print("[3/9] validate contracts, render dbt model contracts")
    validation = {d: C.validate(C.current(d)) for d in DOMAINS}
    assert not any(validation.values()), validation
    W.render_contract_yaml()

    print("[4/9] dbt build: orders, payments, logistics, then the analytics consumer")
    builds = {}
    for p in DBT_PROJECTS:
        b = W.dbt_build(p)
        if not b["success"]:
            raise SystemExit(f"dbt build failed for {p}: {b}")
        statuses = [x["status"] for x in b["results"]]
        builds[p] = {
            "models": sum(x["node"].startswith("model.") for x in b["results"]),
            "tests": sum(x["node"].startswith("test.") for x in b["results"]),
            "tests_passed": sum(x["node"].startswith("test.") and s == "pass" for x, s in zip(b["results"], statuses, strict=True)),
        }
    print("      deliberate contract break: the dbt error below is EXPECTED")
    demo = dbt_contract_break_demo()
    (RESULTS / "dbt_contract_break_demo.json").write_text(json.dumps(demo, indent=2))

    print("[5/9] evaluate SLOs")
    slo_reports = [slo.evaluate(C.current(d)) for d in DOMAINS]
    (RESULTS / "slo_status.json").write_text(json.dumps(slo_reports, indent=2, default=str))
    _csv(
        RESULTS / "slo_status.csv",
        [{"domain": r["domain"], "data_product": r["data_product"], **c} for r in slo_reports for c in r["checks"]],
    )

    print("[6/9] OpenLineage events + column-level lineage graph")
    schema = W.catalog_schema()
    events = L.build_events({p: W.manifest(p) for p in DBT_PROJECTS}, schema, slo_reports)
    ev_path = RESULTS / "lineage" / "openlineage_events.jsonl"
    n_events = L.write_events(events, ev_path)
    ev = L.read_events(ev_path)
    g_direct = G.build(ev)
    g_all = G.build(ev, include_indirect=True)
    dg = G.dataset_graph(ev)
    tr_direct = G.trace(g_direct, *GOLD)
    tr_all = G.trace(g_all, *GOLD)
    tr_gmv = G.trace(g_direct, "analytics.main.mart_daily_kpis", "gmv")
    tr_otd = G.trace(g_direct, "analytics.main.mart_daily_kpis", "on_time_delivery_rate")
    G.render_png(
        [tr_direct],
        RESULTS / "figures" / "lineage_net_revenue.png",
        "Column lineage of gold metric mart_daily_kpis.net_revenue (direct, from OpenLineage events; SYNTHETIC)",
    )
    G.render_png(
        [tr_direct, tr_gmv, tr_otd],
        RESULTS / "figures" / "lineage_gold_metrics.png",
        "Direct column lineage of the three gold metrics (from OpenLineage events; SYNTHETIC)",
    )
    G.render_dataset_png(dg, RESULTS / "figures" / "lineage_datasets.png")
    strip = lambda t: {k: v for k, v in t.items() if k != "subgraph"}  # noqa: E731
    traces = {
        "net_revenue_direct": strip(tr_direct),
        "net_revenue_with_indirect": strip(tr_all),
        "gmv_direct": strip(tr_gmv),
        "on_time_delivery_rate_direct": strip(tr_otd),
    }
    (RESULTS / "lineage" / "gold_metric_traces.json").write_text(json.dumps(traces, indent=2))
    (RESULTS / "lineage" / "net_revenue_trace.md").write_text(
        "# Column lineage: `analytics.main.mart_daily_kpis.net_revenue`\n\n"
        "Reconstructed only from `openlineage_events.jsonl` (direct column lineage). SYNTHETIC data.\n\n"
        f"Source columns: {', '.join('`' + s + '`' for s in tr_direct['source_columns'])}\n\n"
        "```mermaid\n" + G.mermaid(tr_direct["subgraph"]) + "\n```\n"
    )

    print("[7/9] CI gate: replay release history + labelled change scenarios")
    hist = scenarios.replay_history()
    suite = scenarios.run_suite()
    _csv(RESULTS / "contract_gate_history.csv", hist)
    _csv(RESULTS / "contract_gate_scenarios.csv", suite)

    print("[8/9] consumer breakage replay (naive raw reads vs versioned data products)")
    rep = consumers.replay()
    raw_changes = consumers.raw_change_summary()
    _csv(RESULTS / "consumer_replay.csv", rep["dashboards"])
    _csv(RESULTS / "raw_feed_changes.csv", raw_changes)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = [r["dashboard"] for r in rep["dashboards"]]
    x = range(len(names))
    ax.bar(
        [i - 0.2 for i in x], [r["naive_broken_days"] for r in rep["dashboards"]], 0.4, label="reads raw feed (pre-mesh)", color="#c0504d"
    )
    ax.bar(
        [i + 0.2 for i in x], [r["mesh_broken_days"] for r in rep["dashboards"]], 0.4, label="reads versioned data product", color="#2a6fdb"
    )
    ax.set_xticks(list(x), names, fontsize=8)
    ax.set_ylabel("days broken (of 90)")
    ax.set_title("Dashboard breakage over 90 days of upstream schema change (SYNTHETIC)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "consumer_breakage.png", dpi=130)
    plt.close(fig)

    print("[9/9] catalogue, deprecation report, gold KPIs, metrics")
    md, html = catalog.render(slo_reports, dg)
    (RESULTS / "catalog.md").write_text(md)
    (RESULTS / "catalog.html").write_text(html)
    dep = catalog.deprecation_report(dg)
    (RESULTS / "deprecation_report.json").write_text(json.dumps(dep, indent=2))

    con = duckdb.connect(str(WAREHOUSE / "analytics.duckdb"), read_only=True)
    for d in DOMAINS:  # the consumer's views read the domains' attached databases
        con.execute(f"attach '{WAREHOUSE / (d + '.duckdb')}' as {d} (read_only)")
    k = con.execute(
        "select count(*), sum(orders), sum(gmv), sum(net_revenue), sum(on_time_delivery_rate * orders) / sum(orders) from main.mart_daily_kpis"
    ).fetchone()
    otd = con.execute("select avg(case when delivered_on_time then 1.0 else 0.0 end), count(*) from main.int_order_fulfilment").fetchone()
    legacy = con.execute("select sum(completed_revenue) from main.rpt_legacy_revenue").fetchone()[0]
    con.close()

    n_models = sum(b["models"] for b in builds.values())
    n_tests = sum(b["tests"] for b in builds.values())
    n_tests_ok = sum(b["tests_passed"] for b in builds.values())
    metrics = {
        "data": {
            "path_used": data_info["path_used"],
            "label": "SYNTHETIC",
            "seed": SEED,
            "start": str(START.date()),
            "days": N_DAYS,
            "as_of": str(AS_OF),
            "raw_rows": {d: raw[d]["rows"] for d in DOMAINS},
            "raw_files": {d: raw[d]["files"] for d in DOMAINS},
            "raw_schema_changes": len(raw_changes),
            "raw_schema_changes_removing_columns": sum(r["breaking_for_raw_readers"] for r in raw_changes),
            "sample_bytes": data_info["sample_bytes"],
        },
        "contracts": {
            d: {
                "version": C.current(d)["version"],
                "releases": len(C.history(d)),
                "interfaces": [p for p, _, _ in C.published_interfaces(C.current(d))],
            }
            for d in DOMAINS
        },
        "dbt": {
            "projects": builds,
            "models": n_models,
            "tests": n_tests,
            "tests_passed": n_tests_ok,
            "contract_break_demo_blocked": not demo["dbt_build_success"],
        },
        "ci_gate": {
            "history_transitions": len(hist),
            "history_passed": sum(h["passed"] for h in hist),
            "scenarios": len(suite),
            "scenarios_correct": sum(s["correct"] for s in suite),
            "breaking_scenarios": sum(s["expected"] == "fail" for s in suite),
            "breaking_scenarios_blocked": sum(s["expected"] == "fail" and s["gate"] == "fail" for s in suite),
            "safe_scenarios": sum(s["expected"] == "pass" for s in suite),
            "safe_scenarios_passed": sum(s["expected"] == "pass" and s["gate"] == "pass" for s in suite),
        },
        "slo": {
            r["domain"]: {
                "status": r["status"],
                "passed": r["passed"],
                "total": r["total"],
                **{c["sla"]: {"observed": c["observed"], "target": c["target"], "status": c["status"]} for c in r["checks"]},
            }
            for r in slo_reports
        },
        "lineage": {
            "openlineage_events": n_events,
            "jobs": len({e["job"]["name"] for e in ev}),
            "column_nodes": g_direct.number_of_nodes(),
            "column_edges_direct": g_direct.number_of_edges(),
            "column_edges_with_indirect": g_all.number_of_edges(),
            "gold_metric": f"{GOLD[0]}.{GOLD[1]}",
            "net_revenue_source_columns_direct": tr_direct["source_columns"],
            "net_revenue_upstream_columns_direct": tr_direct["upstream_columns"],
            "net_revenue_hops": tr_direct["max_hops"],
            "net_revenue_jobs": len(tr_direct["jobs"]),
            "net_revenue_source_columns_with_indirect": len(tr_all["source_columns"]),
            "net_revenue_domains_with_indirect": tr_all["producer_domains"],
        },
        "consumers": {k2: v for k2, v in rep.items() if k2 != "dashboards"},
        "deprecations": dep,
        "gold_kpis": {
            "days": int(k[0]),
            "orders": int(k[1]),
            "gmv": float(k[2]),
            "net_revenue": float(k[3]),
            "net_to_gmv": round(float(k[3]) / float(k[2]), 4),
            "on_time_delivery_rate": round(float(otd[0]), 4),
            "shipped_orders": int(otd[1]),
            "legacy_v1_completed_revenue": float(legacy),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (RESULTS / "report.md").write_text(report_md(metrics, hist, suite, rep, slo_reports, demo))
    print(
        json.dumps(
            {"runtime_seconds": metrics["runtime_seconds"], "ci_gate": metrics["ci_gate"], "consumers": metrics["consumers"]}, indent=1
        )
    )


def report_md(m: dict, hist: list, suite: list, rep: dict, slo_reports: list, demo: dict) -> str:
    L_ = [
        "# Day 19 run report (SYNTHETIC data)",
        "",
        f"As of {m['data']['as_of']}; seed {m['data']['seed']}; runtime {m['runtime_seconds']} s.",
        "",
    ]
    L_ += [
        "## SLO status",
        "",
        "| product | version | freshness h | completeness % | uniqueness % | rows | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in slo_reports:
        c = {x["sla"]: x for x in r["checks"]}
        cell = lambda s, c=c: f"{c[s]['observed']} (target {c[s]['target']}) {c[s]['status']}"  # noqa: E731
        L_ += [
            f"| {r['data_product']} | {r['contract_version']} | {cell('freshness')} | {cell('completeness')} | {cell('uniqueness')} | {cell('volume')} | **{r['status']}** |"
        ]
    L_ += [
        "",
        "## Contract release history through the gate",
        "",
        "| domain | version | required | actual | passed |",
        "|---|---|---|---|---|",
    ]
    for h in hist:
        L_ += [f"| {h['domain']} | {h['version']} | {h['required_bump']} | {h['actual_bump']} | {h['passed']} |"]
    L_ += [
        "",
        "## Labelled change scenarios",
        "",
        "| scenario | version | expected | gate | correct | first error |",
        "|---|---|---|---|---|---|",
    ]
    for s in suite:
        L_ += [f"| {s['scenario']} | {s['version']} | {s['expected']} | {s['gate']} | {s['correct']} | {s['first_error'][:110]} |"]
    L_ += [
        "",
        "## dbt enforced-contract demo",
        "",
        f"- change: {demo['change']}",
        f"- dbt build succeeded: {demo['dbt_build_success']}",
        f"- message: `{demo['message'][:300]}`",
    ]
    L_ += [
        "",
        "## Consumer breakage replay",
        "",
        "| dashboard | domain | live days | broken days (raw) | first break | broken days (mesh) | interface |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rep["dashboards"]:
        L_ += [
            f"| {r['dashboard']} | {r['domain']} | {r['live_days']} | {r['naive_broken_days']} | {r['naive_first_break'] or '-'} | {r['mesh_broken_days']} | {r['mesh_interface']} |"
        ]
    li = m["lineage"]
    L_ += [
        "",
        "## Lineage",
        "",
        f"- {li['openlineage_events']} OpenLineage events, {li['jobs']} jobs, {li['column_nodes']} column nodes, {li['column_edges_direct']} direct edges",
    ]
    L_ += [
        f"- `{li['gold_metric']}` <- {', '.join(li['net_revenue_source_columns_direct'])} ({li['net_revenue_hops']} hops, {li['net_revenue_jobs']} jobs)"
    ]
    L_ += [
        f"- incl. indirect (join/filter/group-by): {li['net_revenue_source_columns_with_indirect']} source columns from {', '.join(li['net_revenue_domains_with_indirect'])}"
    ]
    g = m["gold_kpis"]
    L_ += [
        "",
        "## Gold KPIs (SYNTHETIC)",
        "",
        f"- {g['orders']} orders over {g['days']} days; GMV {g['gmv']:,.2f}; net revenue {g['net_revenue']:,.2f} ({g['net_to_gmv']:.1%} of GMV)",
    ]
    L_ += [f"- on-time delivery {g['on_time_delivery_rate']:.1%} of {g['shipped_orders']} shipped orders"]
    return "\n".join(L_) + "\n"


if __name__ == "__main__":
    main()
