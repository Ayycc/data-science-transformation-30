"""Temporal multigraph construction and summary statistics."""

from __future__ import annotations

import igraph as ig
import numpy as np
import pandas as pd

from .config import DAY


def collapse_edges(tx: pd.DataFrame) -> pd.DataFrame:
    """Collapse the temporal multigraph to one weighted edge per ordered (src, dst) pair."""
    g = tx.groupby(["src", "dst"], sort=False).agg(
        n_tx=("amount", "size"), amount=("amount", "sum"), first_ts=("ts", "min"), last_ts=("ts", "max")
    )
    return g.reset_index()


def build_igraph(n_nodes: int, pairs: pd.DataFrame, directed: bool = True) -> ig.Graph:
    """Directed weighted igraph from collapsed pairs (vertex id == account id)."""
    g = ig.Graph(n=n_nodes, edges=list(zip(pairs["src"].to_numpy(), pairs["dst"].to_numpy(), strict=True)), directed=directed)
    g.es["amount"] = pairs["amount"].to_numpy()
    g.es["n_tx"] = pairs["n_tx"].to_numpy()
    return g


def undirected_weighted(n_nodes: int, pairs: pd.DataFrame) -> ig.Graph:
    """Undirected simple graph with weight = number of transactions between the two accounts."""
    a = np.minimum(pairs["src"].to_numpy(), pairs["dst"].to_numpy())
    b = np.maximum(pairs["src"].to_numpy(), pairs["dst"].to_numpy())
    u = pd.DataFrame({"a": a, "b": b, "w": pairs["n_tx"].to_numpy()}).groupby(["a", "b"], sort=False)["w"].sum().reset_index()
    g = ig.Graph(n=n_nodes, edges=list(zip(u["a"].to_numpy(), u["b"].to_numpy(), strict=True)), directed=False)
    g.es["weight"] = u["w"].to_numpy().astype(float)
    return g


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def summarise(tx: pd.DataFrame, pairs: pd.DataFrame, g: ig.Graph) -> dict:
    """Headline structure of the temporal transaction graph."""
    n, m = g.vcount(), g.ecount()
    deg_in = np.array(g.indegree())
    deg_out = np.array(g.outdegree())
    wcc = g.connected_components(mode="weak")
    scc = g.connected_components(mode="strong")
    span_days = float((tx["ts"].max() - tx["ts"].min()) / DAY)
    return {
        "n_accounts": int(n),
        "n_transactions": int(len(tx)),
        "n_unique_pairs": int(m),
        "span_days": round(span_days, 1),
        "total_amount": round(float(tx["amount"].sum()), 2),
        "density": float(m / (n * (n - 1))),
        "reciprocity": float(g.reciprocity(ignore_loops=True, mode="default")),
        "mean_in_degree": float(deg_in.mean()),
        "max_in_degree": int(deg_in.max()),
        "max_out_degree": int(deg_out.max()),
        "gini_in_degree": gini(deg_in),
        "largest_wcc_share": float(max(wcc.sizes()) / n),
        "largest_scc_share": float(max(scc.sizes()) / n),
        "n_scc_nontrivial": int(sum(1 for s in scc.sizes() if s > 1)),
        "isolated_accounts": int(((deg_in + deg_out) == 0).sum()),
    }


def weekly_snapshots(tx: pd.DataFrame) -> pd.DataFrame:
    """Temporal graph: per-week active accounts, edges, new counterparty pairs and laundering share."""
    week = (tx["ts"] // (7 * DAY)).astype(int)
    df = tx.assign(week=week)
    first_seen = df.groupby(["src", "dst"])["week"].transform("min")
    df = df.assign(new_pair=first_seen == df["week"])
    rows = []
    for w, d in df.groupby("week"):
        active = np.union1d(d["src"].to_numpy(), d["dst"].to_numpy())
        rows.append(
            {
                "week": int(w),
                "transactions": int(len(d)),
                "active_accounts": int(len(active)),
                "unique_pairs": int(d[["src", "dst"]].drop_duplicates().shape[0]),
                "new_pair_share": float(d["new_pair"].mean()),
                "amount": float(d["amount"].sum()),
                "laundering_tx": int(d["is_laundering"].sum()) if "is_laundering" in d else 0,
            }
        )
    return pd.DataFrame(rows)
