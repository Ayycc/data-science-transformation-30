"""Build a column-level lineage graph *from the OpenLineage events alone*, trace metrics, and render it."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402


def node_id(dataset: str, field: str) -> str:
    return f"{dataset}::{field}"


def build(events: list[dict[str, Any]], include_indirect: bool = False) -> nx.DiGraph:
    """Column graph. Direct edges come from ``columnLineage.fields``; with ``include_indirect`` the
    dataset-level INDIRECT inputs (join/filter/group-by/window columns) feed every output column."""
    g = nx.DiGraph()
    for e in events:
        if e.get("eventType") != "COMPLETE":
            continue
        job = e["job"]["name"]
        for o in e.get("outputs", []):
            fields = ((o.get("facets") or {}).get("columnLineage") or {}).get("fields", {})
            for col, spec in fields.items():
                dst = node_id(o["name"], col)
                g.add_node(dst, dataset=o["name"], field=col)
                for inp in spec["inputFields"]:
                    src = node_id(inp["name"], inp["field"])
                    g.add_node(src, dataset=inp["name"], field=inp["field"])
                    sub = (inp.get("transformations") or [{}])[0].get("subtype")
                    g.add_edge(src, dst, job=job, subtype=sub, type="DIRECT")
            if include_indirect:
                cols = [f["name"] for f in ((o.get("facets") or {}).get("schema") or {}).get("fields", [])]
                for inp in ((o.get("facets") or {}).get("columnLineage") or {}).get("dataset", []) or []:
                    src = node_id(inp["name"], inp["field"])
                    g.add_node(src, dataset=inp["name"], field=inp["field"])
                    sub = (inp.get("transformations") or [{}])[0].get("subtype")
                    for col in cols:
                        dst = node_id(o["name"], col)
                        g.add_node(dst, dataset=o["name"], field=col)
                        if not g.has_edge(src, dst):
                            g.add_edge(src, dst, job=job, subtype=sub, type="INDIRECT")
    return g


def dataset_graph(events: list[dict[str, Any]]) -> nx.DiGraph:
    g = nx.DiGraph()
    for e in events:
        if e.get("eventType") != "COMPLETE":
            continue
        for i in e.get("inputs", []):
            for o in e.get("outputs", []):
                g.add_edge(i["name"], o["name"], job=e["job"]["name"])
    return g


def trace(g: nx.DiGraph, dataset: str, field: str) -> dict[str, Any]:
    target = node_id(dataset, field)
    if target not in g:
        raise KeyError(f"{target} has no lineage in the events")
    up = nx.ancestors(g, target) | {target}
    sub = g.subgraph(up).copy()
    sources = sorted(n for n in sub if sub.in_degree(n) == 0)
    longest = nx.dag_longest_path_length(sub)
    jobs = sorted({d["job"] for _, _, d in sub.edges(data=True)})
    domains = sorted({_domain(sub.nodes[n]["dataset"]) for n in sub} - {"analytics"})
    return {
        "metric": target,
        "upstream_columns": len(up) - 1,
        "source_columns": sources,
        "datasets": sorted({sub.nodes[n]["dataset"] for n in sub}),
        "jobs": jobs,
        "max_hops": longest,
        "producer_domains": domains,
        "subgraph": sub,
    }


def _domain(ds: str) -> str:
    return ds.split("/")[0] if ds.endswith(".csv") else ds.split(".")[0]


def _short(ds: str) -> str:
    if ds.endswith(".csv"):
        return "file:" + ds.split("/")[0] + "/*.csv"
    return ds.replace(".main.", ".")


def _layers(sub: nx.DiGraph) -> dict[str, int]:
    depth: dict[str, int] = {}
    for n in nx.topological_sort(sub):
        preds = list(sub.predecessors(n))
        depth[n] = 0 if not preds else 1 + max(depth[p] for p in preds)
    return depth


def render_png(traces: list[dict[str, Any]], path: Path, title: str) -> None:
    sub = nx.compose_all([t["subgraph"] for t in traces])
    depth = _layers(sub)
    last = max(depth.values())
    depth.update({n: last for n in sub if sub.out_degree(n) == 0})  # gold metrics share the right-most column
    # group by dataset within each layer so columns of the same table sit together
    by_layer: dict[int, list[str]] = defaultdict(list)
    for n, d in depth.items():
        by_layer[d].append(n)
    pos = {}
    for d, ns in by_layer.items():
        ns.sort(key=lambda n: (sub.nodes[n]["dataset"], sub.nodes[n]["field"]))
        for i, n in enumerate(ns):
            pos[n] = (d * 3.2, -(i - (len(ns) - 1) / 2) * 0.62)
    domain_color = {"orders": "#2a6fdb", "payments": "#1f9e74", "logistics": "#d9822b", "analytics": "#7b4fd0"}
    colors = []
    for n in sub:
        colors.append(domain_color.get(_domain(sub.nodes[n]["dataset"]), "#888888"))
    height = max(6.0, 0.36 * max(len(v) for v in by_layer.values()) + 2)
    fig, ax = plt.subplots(figsize=(3.3 * (max(by_layer) + 1) + 2, height))
    nx.draw_networkx_edges(sub, pos, ax=ax, edge_color="#9aa3ad", arrows=True, arrowsize=8, width=0.8, node_size=60)
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=colors, node_size=60)
    labels = {n: f"{_short(sub.nodes[n]['dataset'])}.{sub.nodes[n]['field']}" for n in sub}
    for n, (x, y) in pos.items():
        ax.text(x, y + 0.16, labels[n], fontsize=6.5, ha="center", va="bottom")
    for dom, c in domain_color.items():
        ax.scatter([], [], color=c, label=dom)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_dataset_png(g: nx.DiGraph, path: Path) -> None:
    depth = _layers(g)
    by_layer: dict[int, list[str]] = defaultdict(list)
    for n, d in depth.items():
        by_layer[d].append(n)
    pos = {}
    for d, ns in by_layer.items():
        ns.sort()
        for i, n in enumerate(ns):
            pos[n] = (d * 3.0, -(i - (len(ns) - 1) / 2) * 1.0)
    fig, ax = plt.subplots(figsize=(3.0 * (max(by_layer) + 1) + 2, max(5, 0.9 * max(len(v) for v in by_layer.values()) + 1)))
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#9aa3ad", arrows=True, arrowsize=10, node_size=200)
    domain_color = {"orders": "#2a6fdb", "payments": "#1f9e74", "logistics": "#d9822b", "analytics": "#7b4fd0"}
    for n, (x, y) in pos.items():
        ax.scatter([x], [y], color=domain_color.get(_domain(n), "#888"), s=140, zorder=3)
        ax.text(x, y + 0.22, _short(n), fontsize=7.5, ha="center", va="bottom")
    ax.set_title("Dataset-level lineage reconstructed from OpenLineage events (SYNTHETIC data)", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def mermaid(sub: nx.DiGraph) -> str:
    ids = {n: f"n{i}" for i, n in enumerate(sorted(sub))}
    lines = ["flowchart LR"]
    for n in sorted(sub):
        lines.append(f'    {ids[n]}["{_short(sub.nodes[n]["dataset"])}.{sub.nodes[n]["field"]}"]')
    for a, b, d in sorted(sub.edges(data=True)):
        lines.append(f"    {ids[a]} -->|{d.get('subtype') or ''}| {ids[b]}")
    return "\n".join(lines)
