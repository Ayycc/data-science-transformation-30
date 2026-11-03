"""In-process ANN serving with FAISS (HNSW, inner product) and request-level latency measurement."""

from __future__ import annotations

import time

import faiss
import numpy as np


def build_hnsw(item_emb: np.ndarray, m: int = 32, ef_construction: int = 200, ef_search: int = 128) -> faiss.Index:
    index = faiss.IndexHNSWFlat(item_emb.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(np.ascontiguousarray(item_emb, dtype=np.float32))
    index.hnsw.efSearch = ef_search
    return index


def build_flat(item_emb: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(item_emb.shape[1])
    index.add(np.ascontiguousarray(item_emb, dtype=np.float32))
    return index


def ann_recall(hnsw: faiss.Index, flat: faiss.Index, queries: np.ndarray, k: int = 100) -> float:
    """Share of the exact top-``k`` neighbours that HNSW also returns."""
    _, exact = flat.search(queries, k)
    _, approx = hnsw.search(queries, k)
    return float(np.mean([len(np.intersect1d(a, e)) / k for a, e in zip(approx, exact, strict=True)]))


def search_filtered(index: faiss.Index, queries: np.ndarray, seen_lists: list[np.ndarray], k: int, overfetch: int) -> np.ndarray:
    """ANN top-``k`` per query after removing already-watched items (fetches ``k + overfetch`` then filters)."""
    ef = index.hnsw.efSearch if hasattr(index, "hnsw") else None
    if ef is not None:
        index.hnsw.efSearch = max(ef, k + overfetch)
    _, ids = index.search(np.ascontiguousarray(queries, dtype=np.float32), k + overfetch)
    if ef is not None:
        index.hnsw.efSearch = ef  # restore the serving setting
    out = np.empty((len(queries), k), dtype=np.int64)
    for r, (row, seen) in enumerate(zip(ids, seen_lists, strict=True)):
        row = row[row >= 0]
        row = row[~np.isin(row, seen)]
        if len(row) < k:  # extremely heavy users: pad with exact search on the remaining ids (rare)
            row = np.concatenate([row, np.setdiff1d(np.arange(index.ntotal), np.concatenate([row, seen]))[: k - len(row)]])
        out[r] = row[:k]
    return out


def measure_latency(user_fn, index: faiss.Index, users: np.ndarray, k: int = 100, warmup: int = 50) -> dict:
    """Single-request latency (one user per call, one thread): user-tower forward + ANN search."""
    n_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    tower, search, total = [], [], []
    for j, u in enumerate(users):
        t0 = time.perf_counter()
        q = user_fn(int(u))
        t1 = time.perf_counter()
        index.search(q, k)
        t2 = time.perf_counter()
        if j >= warmup:
            tower.append(t1 - t0)
            search.append(t2 - t1)
            total.append(t2 - t0)
    faiss.omp_set_num_threads(n_threads)

    def pct(x: list[float]) -> dict:
        a = np.asarray(x) * 1000.0
        return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99))}

    return {
        "n_requests": len(total),
        "user_tower": pct(tower),
        "ann_search": pct(search),
        "end_to_end": pct(total),
        "samples_ms": (np.asarray(total) * 1000).tolist(),
    }
