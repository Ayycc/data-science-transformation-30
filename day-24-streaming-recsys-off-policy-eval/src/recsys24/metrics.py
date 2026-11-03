"""Offline ranking metrics: recall@k, NDCG@k, coverage, novelty, popularity bias and intra-list diversity."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def interaction_matrix(users: np.ndarray, items: np.ndarray, n_users: int, n_items: int) -> sp.csr_matrix:
    m = sp.csr_matrix((np.ones(len(users), dtype=np.float32), (users, items)), shape=(n_users, n_items))
    m.data[:] = 1.0  # binarise duplicates
    return m


def top_k(scores: np.ndarray, seen: sp.csr_matrix | None, k: int) -> np.ndarray:
    """Top-``k`` item ids per row of a dense score matrix, excluding already-seen items."""
    s = np.array(scores, dtype=np.float32, copy=True)
    if seen is not None:
        r, c = seen.nonzero()
        s[r, c] = -np.inf
    part = np.argpartition(-s, kth=min(k, s.shape[1] - 1), axis=1)[:, :k]
    order = np.take_along_axis(s, part, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(part, order, 1)


def _hits(recs: np.ndarray, truth: sp.csr_matrix) -> np.ndarray:
    rows = np.arange(recs.shape[0])[:, None]
    return np.asarray(truth[rows, recs].todense() > 0)


def recall_at_k(recs: np.ndarray, truth: sp.csr_matrix, k: int) -> float:
    h = _hits(recs[:, :k], truth)
    n_true = np.asarray(truth.sum(1)).ravel()
    ok = n_true > 0
    return float((h.sum(1)[ok] / n_true[ok]).mean())


def ndcg_at_k(recs: np.ndarray, truth: sp.csr_matrix, k: int) -> float:
    h = _hits(recs[:, :k], truth).astype(float)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (h * disc).sum(1)
    n_true = np.asarray(truth.sum(1)).ravel().astype(int)
    idcg = np.array([disc[: min(n, k)].sum() for n in n_true])
    ok = n_true > 0
    return float((dcg[ok] / idcg[ok]).mean())


def per_user_ndcg(recs: np.ndarray, truth: sp.csr_matrix, k: int) -> np.ndarray:
    h = _hits(recs[:, :k], truth).astype(float)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    n_true = np.asarray(truth.sum(1)).ravel().astype(int)
    idcg = np.array([disc[: min(n, k)].sum() if n else 1.0 for n in n_true])
    return (h * disc).sum(1) / idcg


def coverage(recs: np.ndarray, n_items: int) -> float:
    return float(len(np.unique(recs)) / n_items)


def novelty(recs: np.ndarray, item_pop: np.ndarray, n_users: int) -> float:
    """Mean self-information -log2(p_i) of recommended items, p_i = share of users who watched i."""
    p = (item_pop + 1.0) / (n_users + 1.0)
    return float((-np.log2(p[recs])).mean())


def popularity_profile(recs: np.ndarray, item_pop: np.ndarray, head_frac: float = 0.1) -> dict:
    """Average recommendation popularity (ARP) and share of recommendations from the head of the catalogue."""
    order = np.argsort(-item_pop)
    head = np.zeros(len(item_pop), dtype=bool)
    head[order[: int(round(head_frac * len(item_pop)))]] = True
    return {"arp": float(item_pop[recs].mean()), "head_share": float(head[recs].mean())}


def intra_list_diversity(recs: np.ndarray, item_genres: np.ndarray) -> float:
    """1 - mean pairwise cosine similarity of genre vectors within each list."""
    g = item_genres / np.linalg.norm(item_genres, axis=1, keepdims=True).clip(1e-9)
    v = g[recs]  # n x k x G
    sim = np.einsum("nkg,nlg->nkl", v, v)
    k = recs.shape[1]
    off = (sim.sum((1, 2)) - np.trace(sim, axis1=1, axis2=2)) / (k * (k - 1))
    return float(1.0 - off.mean())


def evaluate_recs(recs: np.ndarray, truth: sp.csr_matrix, item_pop: np.ndarray, item_genres: np.ndarray, n_users: int) -> dict:
    out = {
        "recall@10": recall_at_k(recs, truth, 10),
        "ndcg@10": ndcg_at_k(recs, truth, 10),
        "coverage@10": coverage(recs[:, :10], truth.shape[1]),
        "novelty@10": novelty(recs[:, :10], item_pop, n_users),
        "ild@10": intra_list_diversity(recs[:, :10], item_genres),
    }
    if recs.shape[1] >= 50:
        out["recall@50"] = recall_at_k(recs, truth, 50)
    prof = popularity_profile(recs[:, :10], item_pop)
    out["arp@10"] = prof["arp"]
    out["head_share@10"] = prof["head_share"]
    return out
