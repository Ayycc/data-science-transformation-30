from __future__ import annotations

import numpy as np

from aml_graph.config import DAY, SMALL, TYPOLOGIES
from aml_graph.graph import build_igraph, collapse_edges, gini, summarise, weekly_snapshots
from aml_graph.simulate import simulate


def test_deterministic(small_bank):
    again = simulate(SMALL)
    assert small_bank.transactions.equals(again.transactions)


def test_typologies_and_labels(small_bank):
    tx, acc, inst = small_bank.transactions, small_bank.accounts, small_bank.instances
    assert set(inst["typology"]) == set(TYPOLOGIES)
    assert inst.groupby("typology")["instance_id"].nunique().eq(SMALL.instances_per_typology).all()
    lt = tx[tx["is_laundering"]]
    flagged = set(lt["src"]) | set(lt["dst"])
    assert flagged == set(acc.loc[acc["is_laundering"], "account_id"])
    assert (tx["src"] != tx["dst"]).all()
    assert tx["ts"].between(0, SMALL.n_days * DAY).all()
    assert tx["ts"].is_monotonic_increasing


def test_background_respects_open_dates(small_bank):
    tx, acc = small_bank.transactions, small_bank.accounts
    bg = tx[~tx["is_laundering"]]
    open_s = acc["open_day"].to_numpy() * DAY
    assert (bg["ts"].to_numpy() >= open_s[bg["src"].to_numpy()] - 1e-6).all()


def test_summary_and_snapshots(small_bank):
    tx = small_bank.transactions
    pairs = collapse_edges(tx)
    g = build_igraph(len(small_bank.accounts), pairs)
    s = summarise(tx, pairs, g)
    assert s["n_transactions"] == len(tx) and s["n_unique_pairs"] == len(pairs)
    assert 0 < s["largest_wcc_share"] <= 1
    snap = weekly_snapshots(tx)
    assert snap["transactions"].sum() == len(tx)
    assert snap["laundering_tx"].sum() == tx["is_laundering"].sum()


def test_gini():
    assert gini(np.ones(10)) == 0.0
    assert gini(np.array([0, 0, 0, 10])) > 0.7
