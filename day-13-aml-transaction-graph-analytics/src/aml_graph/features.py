"""Account-level features.

* **tabular**: what a non-graph monitoring system sees for one account in isolation:
  KYC attributes and aggregates of the account's own transactions (counts, sums, amount
  shape, structuring-band share, night share, active days).
* **structural**: PageRank, sampled betweenness, distinct counterparties, reciprocity,
  clustering, k-core, community context.
* **temporal_flow**: flow-through ratio, burstiness, flow-link (pass-through) counts and
  hold times, max activity in 72 h windows.
* **motif**: counts of label-free typology motifs the account takes part in.
* **neighbour**: GraphSAGE-style (non-learned) mean aggregation of neighbours' features.
"""

from __future__ import annotations

import igraph as ig
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import DAY, HOUR


def tabular_features(acc: pd.DataFrame, tx: pd.DataFrame, horizon_s: float) -> pd.DataFrame:
    n = len(acc)
    idx = pd.RangeIndex(n, name="account_id")
    f = pd.DataFrame(index=idx)
    f["role_business"] = (acc["role"] == "business").astype(float).to_numpy()
    f["role_merchant"] = (acc["role"] == "merchant").astype(float).to_numpy()
    f["region_risk"] = acc["region_risk"].astype(float).to_numpy()
    f["account_age_days"] = (horizon_s / DAY - acc["open_day"]).to_numpy()
    amt = tx["amount"].to_numpy()
    hour = (tx["ts"].to_numpy() % DAY) / HOUR
    night = ((hour < 6) | (hour >= 22)).astype(float)
    band = ((amt >= 9000) & (amt < 10000)).astype(float)
    rnd = (np.mod(amt, 100) == 0).astype(float)
    day = (tx["ts"].to_numpy() // DAY).astype(np.int64)
    for side, col in (("out", "src"), ("in", "dst")):
        d = pd.DataFrame(
            {"a": tx[col].to_numpy(), "amt": amt, "night": night, "band": band, "rnd": rnd, "big": (amt >= 1000).astype(float)}
        )
        g = d.groupby("a")
        f[f"n_{side}"] = g.size().reindex(idx, fill_value=0).to_numpy()
        f[f"sum_{side}"] = g["amt"].sum().reindex(idx, fill_value=0).to_numpy()
        f[f"mean_{side}"] = g["amt"].mean().reindex(idx, fill_value=0).to_numpy()
        f[f"max_{side}"] = g["amt"].max().reindex(idx, fill_value=0).to_numpy()
        f[f"std_{side}"] = g["amt"].std().reindex(idx).fillna(0).to_numpy()
        f[f"night_share_{side}"] = g["night"].mean().reindex(idx, fill_value=0).to_numpy()
        f[f"band_share_{side}"] = g["band"].mean().reindex(idx, fill_value=0).to_numpy()
        f[f"round_share_{side}"] = g["rnd"].mean().reindex(idx, fill_value=0).to_numpy()
        f[f"n_material_{side}"] = g["big"].sum().reindex(idx, fill_value=0).to_numpy()
    both = pd.DataFrame({"a": np.concatenate([tx["src"].to_numpy(), tx["dst"].to_numpy()]), "d": np.concatenate([day, day])})
    f["active_days"] = both.drop_duplicates().groupby("a").size().reindex(idx, fill_value=0).to_numpy()
    return f


def burstiness(tx: pd.DataFrame, n: int) -> np.ndarray:
    """Goh-Barabasi burstiness B = (sigma - mu) / (sigma + mu) of inter-event times (in and out)."""
    node = np.concatenate([tx["src"].to_numpy(), tx["dst"].to_numpy()])
    ts = np.concatenate([tx["ts"].to_numpy(), tx["ts"].to_numpy()])
    order = np.lexsort((ts, node))
    node, ts = node[order], ts[order]
    gap = np.diff(ts)
    same = node[1:] == node[:-1]
    df = pd.DataFrame({"a": node[1:][same], "gap": gap[same]})
    g = df.groupby("a")["gap"]
    mu, sd, cnt = g.mean(), g.std(ddof=0), g.size()
    b = ((sd - mu) / (sd + mu)).where(cnt >= 3)
    return b.reindex(pd.RangeIndex(n)).fillna(0.0).to_numpy()


def structural_features(
    g: ig.Graph, gu: ig.Graph, membership: np.ndarray, motif_any: np.ndarray, n_sources: int, seed: int
) -> pd.DataFrame:
    n = g.vcount()
    f = pd.DataFrame(index=pd.RangeIndex(n, name="account_id"))
    f["indeg_distinct"] = g.indegree()
    f["outdeg_distinct"] = g.outdegree()
    f["pagerank"] = np.array(g.pagerank(weights="amount", directed=True)) * n
    rev = g.copy()
    rev.reverse_edges()
    f["pagerank_rev"] = np.array(rev.pagerank(weights="amount", directed=True)) * n
    rng = np.random.default_rng(seed)
    src = rng.choice(n, size=min(n_sources, n), replace=False).tolist()
    bt = np.array(g.betweenness(directed=True, sources=src))
    f["betweenness_sampled"] = bt * (n / len(src)) / n  # scaled estimate, normalised by n
    # reciprocity: share of distinct counterparties that are both senders and receivers
    succ = [set(s) for s in g.get_adjlist(mode="out")]
    pred = [set(p) for p in g.get_adjlist(mode="in")]
    recip = np.array([len(s & p) / max(1, len(s | p)) for s, p in zip(succ, pred, strict=True)])
    f["reciprocity"] = recip
    f["clustering"] = np.nan_to_num(np.array(gu.transitivity_local_undirected(mode="zero")))
    f["coreness"] = gu.coreness()
    sizes = np.bincount(membership)
    f["community_size"] = sizes[membership]
    nbrs = gu.get_adjlist()
    f["out_of_community_share"] = [np.mean(membership[nb] != membership[i]) if nb else 0.0 for i, nb in enumerate(nbrs)]
    flag_rate = np.bincount(membership, weights=motif_any) / sizes
    f["community_motif_rate"] = flag_rate[membership]
    return f


def temporal_flow_features(tab: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=tab.index)
    s_in, s_out = tab["sum_in"].to_numpy(), tab["sum_out"].to_numpy()
    f["flow_through_ratio"] = np.minimum(s_in, s_out) / np.maximum(1.0, np.maximum(s_in, s_out))
    f["net_flow_share"] = (s_in - s_out) / np.maximum(1.0, s_in + s_out)
    f["burstiness"] = burstiness(tx, len(tab))
    return f


def neighbour_features(gu: ig.Graph, feats: pd.DataFrame, cols: list[str], hops: int = 2) -> pd.DataFrame:
    """Non-learned GraphSAGE-style aggregation: mean of neighbours' features over 1 and 2 hops."""
    n = gu.vcount()
    el = np.array(gu.get_edgelist())
    if len(el) == 0:
        return pd.DataFrame(index=feats.index)
    a = sp.coo_matrix((np.ones(len(el) * 2), (np.r_[el[:, 0], el[:, 1]], np.r_[el[:, 1], el[:, 0]])), shape=(n, n)).tocsr()
    deg = np.asarray(a.sum(axis=1)).ravel()
    p = sp.diags(1.0 / np.maximum(deg, 1)) @ a
    x = feats[cols].to_numpy(dtype=float)
    out = pd.DataFrame(index=feats.index)
    h = x
    for k in range(1, hops + 1):
        h = p @ h
        for j, c in enumerate(cols):
            out[f"nbr{k}_mean_{c}"] = h[:, j]
    nb_max = np.zeros(n)
    m = feats["motif_any"].to_numpy() if "motif_any" in feats else None
    if m is not None:
        nb_max = np.asarray((a.multiply(m[None, :])).sum(axis=1)).ravel()
        out["nbr1_motif_count"] = nb_max
    return out
