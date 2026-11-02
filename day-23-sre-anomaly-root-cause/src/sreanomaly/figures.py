"""Figures for results/figures (all from SYNTHETIC telemetry)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from . import rca  # noqa: E402

BLUE, ORANGE, GREY, RED = "#2b6cb0", "#dd6b20", "#a0aec0", "#c53030"


def _layout(g: nx.DiGraph) -> dict:
    depth = {}
    for n in nx.topological_sort(g):
        preds = list(g.predecessors(n))
        depth[n] = 0 if not preds else max(depth[p] for p in preds) + 1
    layers: dict[int, list[str]] = {}
    for n, d in depth.items():
        layers.setdefault(d, []).append(n)
    pos = {}
    for d, ns in layers.items():
        for i, n in enumerate(sorted(ns)):
            pos[n] = ((i + 1) / (len(ns) + 1), -d)
    return pos


def topology(out: Path, g: nx.DiGraph, tel) -> None:
    counts = tel.incidents["root"].value_counts()
    pos = _layout(g)
    fig, ax = plt.subplots(figsize=(11, 6))
    sizes = [300 + 160 * counts.get(n, 0) for n in g.nodes]
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=GREY, arrows=True, arrowsize=10, node_size=sizes)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=BLUE, node_size=sizes, alpha=0.85)
    nx.draw_networkx_labels(g, pos, ax=ax, labels={n: f"{n}\n({counts.get(n, 0)})" for n in g.nodes}, font_size=7)
    ax.set_title("SYNTHETIC call graph (caller -> callee); node size and (n) = injected incidents rooted there")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "call_graph.png", dpi=120)
    plt.close(fig)


def funnel(out: Path, f: dict) -> None:
    labels = ["raw metric\nalerts", "per-service\nepisodes (ECOD)", "grouped incidents\n(temporal+topology)", "injected\nincidents"]
    vals = [f["raw_metric_alerts"], f["per_service_episodes"], f["grouped_incidents_topological"], f["injected_incidents"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, vals, color=[GREY, BLUE, BLUE, ORANGE])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("count (14-day test period)")
    ax.set_title(f"Alert volume: {f['reduction_raw_to_incidents_pct']}% fewer pages after de-duplication (SYNTHETIC)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "alert_funnel.png", dpi=120)
    plt.close(fig)


def topk(out: Path, rc: dict) -> None:
    methods = ["max_anomaly", "topology_frontier", "earliest_onset_pelt", "causal_pagerank", "ensemble"]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, k in zip(axes, ("top1_detected", "top3_detected")):
        a = [rc[m][k] for m in methods]
        b = [rc[f"{m}+deploy_prior"][k] for m in methods]
        ax.bar(x - 0.2, a, 0.4, label="no prior", color=BLUE)
        ax.bar(x + 0.2, b, 0.4, label="+ deploy prior", color=ORANGE)
        for i, (u, v) in enumerate(zip(a, b)):
            ax.text(i - 0.2, u + 0.01, f"{u:.2f}", ha="center", fontsize=7)
            ax.text(i + 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x, [m.replace("_", "\n") for m in methods], fontsize=8)
        ax.set_title(k.replace("_detected", " accuracy (detected incidents)"))
        ax.set_ylim(0, 1.08)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower left")
    fig.suptitle("Root-cause ranking on injected SYNTHETIC incidents")
    fig.tight_layout()
    fig.savefig(out / "rca_topk.png", dpi=120)
    plt.close(fig)


def delays(out: Path, rows: list[dict]) -> None:
    d = np.array([r["first_alert"] - r["start"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d, bins=np.arange(d.min() - 0.5, d.max() + 1.5, 1), color=BLUE)
    ax.axvline(np.median(d), color=ORANGE, ls="--", label=f"median {np.median(d):.0f} min")
    ax.set_xlabel("minutes from fault injection to first grouped alert (negative = deploy restart blip alerted first)")
    ax.set_ylabel("incidents")
    ax.legend()
    ax.set_title("Detection delay, ECOD + grouping (SYNTHETIC)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "detection_delay.png", dpi=120)
    plt.close(fig)


def example(out: Path, tel, z: np.ndarray, rows: list[dict], g: nx.DiGraph) -> None:
    sig = rca.anomaly_signal(z)
    good = [r for r in rows if len(r["group_services"]) >= 4 and rca.topk_hit(r["scores"]["ensemble"], r["root"], 1)]
    r = good[0] if good else rows[0]
    lo, hi = r["first_alert"] - rca.PRE, r["first_alert"] + rca.POST + 45
    t = np.arange(lo, hi)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    svcs = sorted(r["true_onsets"], key=lambda s: r["true_onsets"][s])[:6]
    cmap = plt.get_cmap("tab10")
    for k, s in enumerate(svcs):
        j = tel.services.index(s)
        ax.plot(t - r["start"], sig[lo:hi, j], color=cmap(k), lw=1.6 if s == r["root"] else 1.0, label=s + (" (root)" if s == r["root"] else ""))
        o = r["onsets"]["pelt"].get(s)
        if o is not None:
            ax.axvline(o - r["start"], color=cmap(k), ls=":", lw=1)
    dep = tel.deploys[(tel.deploys["minute"] >= lo) & (tel.deploys["minute"] < hi)]
    for d in dep.itertuples():
        ax.axvline(d.minute - r["start"], color=RED, lw=2, alpha=0.5)
        ax.text(d.minute - r["start"], ax.get_ylim()[1] * 0.95, f"deploy {d.service}", color=RED, fontsize=7, rotation=90, va="top")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("minutes from injected fault start (dotted = PELT onset)")
    ax.set_ylabel("anomaly signal (mean positive z)")
    ax.set_title(f"Incident {r['incident_id']}: {r['fault']} at {r['root']} (SYNTHETIC)")
    ax.legend(fontsize=7)
    ax = axes[1]
    H = nx.DiGraph()
    H.add_nodes_from(r["candidates"])
    for c, e, w in r["causal_edges"]:
        H.add_edge(c, e, weight=w)
    pos = nx.spring_layout(H, seed=1)
    sc = r["scores"]["ensemble"]
    nx.draw_networkx_nodes(
        H,
        pos,
        ax=ax,
        node_color=[ORANGE if n == r["root"] else BLUE for n in H],
        node_size=[300 + 3000 * sc[n] / max(sc.values()) for n in H],
        alpha=0.85,
    )
    nx.draw_networkx_edges(H, pos, ax=ax, width=[0.5 + 3 * H[u][v]["weight"] for u, v in H.edges], edge_color=GREY, arrows=True, arrowsize=12)
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=7)
    ax.set_title("Causal graph (cause -> effect; width = PC/Granger weight;\nsize = ensemble score; orange = true root)", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "incident_example.png", dpi=120)
    plt.close(fig)


def make_all(out: Path, tel, z, dets, primary, raw, f, rc, rows, g) -> None:
    out.mkdir(parents=True, exist_ok=True)
    topology(out, g, tel)
    funnel(out, f)
    topk(out, rc)
    delays(out, rows)
    example(out, tel, z, rows, g)
