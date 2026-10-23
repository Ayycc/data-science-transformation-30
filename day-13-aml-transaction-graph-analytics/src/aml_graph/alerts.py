"""Investigator view: explainable alert subgraphs exported as interactive HTML (pyvis)."""

from __future__ import annotations

import html
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network

from .config import DAY
from .motifs import Detection

FEATURE_TEXT = {
    "flow_links_through": "money arrived and left again within 72 h, almost intact",
    "motif_mule_chain": "sits on a temporal pass-through chain",
    "motif_cycle": "sits on a temporal cycle (money returns to its origin)",
    "motif_scatter_gather": "part of a scatter-gather structure",
    "motif_fan_in": "part of a fan-in burst with onward transfer",
    "motif_fan_out": "part of a fan-out burst funded just before",
    "flow_through_ratio": "in-flow and out-flow are nearly equal (pass-through account)",
    "account_age_days": "recently opened account",
    "burstiness": "bursty activity pattern",
}


def alert_subgraph(
    account: int, mat: pd.DataFrame, links: pd.DataFrame, detections: list[Detection], max_edges: int = 60
) -> nx.MultiDiGraph:
    """Explainable subgraph: the motifs the account takes part in plus its largest material transfers."""
    motif_tx: set[int] = set()
    for d in detections:
        if account in d.accounts:
            motif_tx |= set(d.tx_ids)
    lk = links[(links["u"] == account) | (links["v"] == account) | (links["w"] == account)]
    flow_tx = set(lk["tx_in"].astype(int)) | set(lk["tx_out"].astype(int))
    key_tx = motif_tx | flow_tx
    inc = mat[(mat["src"] == account) | (mat["dst"] == account)].nlargest(25, "amount")
    chosen = mat[mat["tx_id"].isin(key_tx)]
    if len(chosen) > max_edges:
        chosen = chosen.nlargest(max_edges, "amount")
    edges = pd.concat([chosen, inc]).drop_duplicates("tx_id")
    g = nx.MultiDiGraph(focal=account)
    for r in edges.itertuples(index=False):
        g.add_edge(int(r.src), int(r.dst), tx_id=int(r.tx_id), amount=float(r.amount), ts=float(r.ts), motif=int(r.tx_id) in key_tx)
    if account not in g:
        g.add_node(account)
    return g


def reason_codes(contrib: pd.Series, values: pd.Series, k: int = 5) -> list[dict]:
    top = contrib.sort_values(ascending=False).head(k)
    out = []
    for f, c in top.items():
        if c <= 0:
            continue
        out.append({"feature": f, "value": float(values[f]), "shap": float(c), "meaning": FEATURE_TEXT.get(f, "")})
    return out


def export_html(
    g: nx.MultiDiGraph, path: Path, title: str, reasons: list[dict], fired: list[str], scores: dict[int, float], truth: dict[int, str]
) -> None:
    focal = g.graph["focal"]
    net = Network(height="640px", width="100%", directed=True, cdn_resources="remote", bgcolor="#fcfcfb", font_color="#0b0b0b")
    net.barnes_hut(gravity=-4000, spring_length=140)
    for n in g.nodes:
        s = scores.get(n)
        tip = f"account {n}" + (f"\nmodel score {s:.3f}" if s is not None else "") + f"\nsynthetic ground truth: {truth.get(n, 'none')}"
        color = "#eb6834" if n == focal else ("#2a78d6" if (s is not None and s >= 0.5) else "#9a9893")
        net.add_node(int(n), label=str(n), title=tip, color=color, size=26 if n == focal else 14)
    for u, v, d in g.edges(data=True):
        day = d["ts"] / DAY
        net.add_edge(
            int(u),
            int(v),
            title=f"tx {d['tx_id']}: {d['amount']:,.2f} on day {day:.2f}",
            label=f"{d['amount']:,.0f}",
            color="#e34948" if d["motif"] else "#c3c2b7",
            width=3 if d["motif"] else 1,
        )
    net.write_html(str(path), notebook=False, open_browser=False)
    rows = "".join(
        f"<li><b>{html.escape(r['feature'])}</b> = {r['value']:.3g} (SHAP +{r['shap']:.2f}) {html.escape(r['meaning'])}</li>"
        for r in reasons
    )
    header = (
        '<div style="font-family:system-ui,sans-serif;padding:12px 16px;max-width:960px;color:#0b0b0b;background:#fcfcfb">'
        f"<h2 style='margin:0 0 6px'>{html.escape(title)}</h2>"
        "<p style='margin:4px 0'>SYNTHETIC data. Orange = alerted account, blue = other held-out (test) accounts scored &ge; 0.5, red edges = transfers in a detected motif or 72 h flow link. Hover nodes and edges for details.</p>"
        f"<p style='margin:4px 0'><b>Typology detectors fired:</b> {html.escape(', '.join(fired) if fired else 'none')}</p>"
        f"<p style='margin:4px 0'><b>Top reasons (TreeSHAP):</b></p><ul style='margin:0'>{rows}</ul></div>"
    )
    text = path.read_text(encoding="utf-8").replace("<body>", "<body>" + header, 1)
    path.write_text(text, encoding="utf-8")


def fired_detectors(account: int, det: dict[str, list[Detection]]) -> list[str]:
    return [t for t, ds in det.items() if any(account in d.accounts for d in ds)]


def score_lookup(index: np.ndarray, scores: np.ndarray) -> dict[int, float]:
    return dict(zip(index.astype(int).tolist(), scores.astype(float).tolist(), strict=True))
