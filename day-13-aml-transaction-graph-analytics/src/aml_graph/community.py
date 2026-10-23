"""Community detection (Louvain and Leiden) on the undirected transaction graph."""

from __future__ import annotations

import igraph as ig
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


def detect_communities(gu: ig.Graph, seed: int) -> dict[str, np.ndarray]:
    """Return membership vectors for Louvain (multilevel) and Leiden (modularity objective)."""
    import random

    random.seed(seed)  # igraph uses Python's RNG
    ig.set_random_number_generator(random)
    louvain = gu.community_multilevel(weights="weight")
    random.seed(seed)
    leiden = gu.community_leiden(objective_function="modularity", weights="weight", n_iterations=10)
    return {"louvain": np.array(louvain.membership), "leiden": np.array(leiden.membership)}


def community_report(gu: ig.Graph, memberships: dict[str, np.ndarray], regions: np.ndarray) -> dict:
    out = {}
    for name, mem in memberships.items():
        sizes = np.bincount(mem)
        out[name] = {
            "n_communities": int((sizes > 0).sum()),
            "n_communities_ge_10": int((sizes >= 10).sum()),
            "largest_community_share": float(sizes.max() / len(mem)),
            "modularity": float(gu.modularity(mem.tolist(), weights="weight")),
            "nmi_vs_planted_regions": float(normalized_mutual_info_score(regions, mem)),
        }
    return out


def community_risk(membership: np.ndarray, flagged: np.ndarray, labels: np.ndarray, min_size: int = 10) -> dict:
    """Rank communities by the share of their accounts flagged by label-free motif detectors.

    Labels are used only to *evaluate* the ranking: what share of laundering accounts sit in the
    top 10% of communities ranked by flag density.
    """
    df = pd.DataFrame({"c": membership, "flag": flagged.astype(float), "y": labels.astype(int)})
    agg = df.groupby("c").agg(size=("flag", "size"), flag_rate=("flag", "mean"), pos=("y", "sum"))
    agg = agg[agg["size"] >= min_size].sort_values("flag_rate", ascending=False)
    k = max(1, int(round(0.10 * len(agg))))
    top = agg.head(k)
    total_pos = labels.sum()
    return {
        "communities_ranked": int(len(agg)),
        "top_decile_communities": int(k),
        "top_decile_account_share": float(top["size"].sum() / len(membership)),
        "top_decile_laundering_share": float(top["pos"].sum() / total_pos) if total_pos else 0.0,
        "laundering_rate_top_decile": float(top["pos"].sum() / top["size"].sum()),
        "laundering_rate_overall": float(labels.mean()),
    }


def instance_cohesion(membership: np.ndarray, instances: pd.DataFrame) -> float:
    """Share of laundering instances whose accounts mostly (>= 50%) fall in a single community."""
    inst = instances.assign(c=membership[instances["account_id"].to_numpy()])
    top_share = inst.groupby("instance_id")["c"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    return float((top_share >= 0.5).mean())
