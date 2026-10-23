from __future__ import annotations

import igraph as ig
import numpy as np
import pandas as pd

from aml_graph.community import community_report, detect_communities
from aml_graph.config import ModelConfig
from aml_graph.features import burstiness, neighbour_features, tabular_features
from aml_graph.graph import collapse_edges, undirected_weighted
from aml_graph.model import grouped_split, operating_point


def test_burstiness_regular_vs_bursty():
    regular = pd.DataFrame({"src": [0] * 10, "dst": [9] * 10, "ts": np.arange(10) * 100.0})
    b = burstiness(regular, 10)
    assert np.isclose(b[0], -1.0)
    bursty = pd.DataFrame({"src": [1] * 10, "dst": [8] * 10, "ts": [0, 1, 2, 3, 4, 1000, 1001, 1002, 5000, 5001.0]})
    assert burstiness(bursty, 10)[1] > 0


def test_tabular_counts(small_bank):
    f = tabular_features(small_bank.accounts, small_bank.transactions, float(small_bank.transactions["ts"].max()))
    tx = small_bank.transactions
    assert f["n_out"].sum() == len(tx) and f["n_in"].sum() == len(tx)
    assert np.isclose(f["sum_in"].sum(), tx["amount"].sum())


def test_neighbour_mean():
    g = ig.Graph(n=3, edges=[(0, 1), (1, 2)])
    feats = pd.DataFrame({"x": [1.0, 2.0, 3.0], "motif_any": [1.0, 0.0, 1.0]})
    out = neighbour_features(g, feats, ["x"], hops=1)
    assert np.allclose(out["nbr1_mean_x"], [2.0, 2.0, 2.0])
    assert np.allclose(out["nbr1_motif_count"], [0, 2, 0])


def test_communities_recover_regions(small_bank):
    pairs = collapse_edges(small_bank.transactions)
    gu = undirected_weighted(len(small_bank.accounts), pairs)
    mem = detect_communities(gu, 1)
    rep = community_report(gu, mem, small_bank.accounts["region"].to_numpy())
    assert rep["leiden"]["modularity"] > 0.3
    assert rep["leiden"]["nmi_vs_planted_regions"] > 0.5


def test_grouped_split_keeps_schemes_together(small_bank):
    acc = small_bank.accounts
    tr, te = grouped_split(acc, ModelConfig())
    inst_tr = set(acc.iloc[tr]["instance_id"]) - {-1}
    inst_te = set(acc.iloc[te]["instance_id"]) - {-1}
    assert not inst_tr & inst_te
    assert len(tr) + len(te) == len(acc)


def test_operating_point():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.75, 0.1, 0.1, 0.1, 0.1, 0.1])
    op = operating_point(y, s, 0.75)
    assert op["recall"] >= 0.75 and op["alerts"] == 4 and op["false_positives"] == 1
