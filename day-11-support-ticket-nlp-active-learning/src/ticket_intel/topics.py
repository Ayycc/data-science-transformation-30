"""Unsupervised topic discovery: dense vectors -> UMAP -> HDBSCAN -> class-based TF-IDF keywords."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

STOP_EXTRA = [
    "__name__",
    "__email__",
    "__phone__",
    "__id__",
    "__amount__",
    "__num__",
    "need",
    "help",
    "want",
    "assistance",
    "know",
    "like",
    "ya",
    "u",
    "im",
]


def umap_reduce(Z: np.ndarray, n_components: int, n_neighbors: int, seed: int) -> np.ndarray:
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    import umap  # imported lazily: numba JIT start-up is slow

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(n_neighbors, len(Z) - 1),
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
            n_jobs=1,
        )
        return reducer.fit_transform(Z)


def cluster(U: np.ndarray, min_cluster_size: int) -> np.ndarray:
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5).fit_predict(U)


def ctfidf_keywords(texts: Sequence[str], labels: np.ndarray, top_n: int = 6) -> dict[int, list[str]]:
    """Class-based TF-IDF (BERTopic-style): one 'document' per cluster, terms weighted against all clusters."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stop = list(ENGLISH_STOP_WORDS.union(STOP_EXTRA))
    clusters = sorted(c for c in set(labels.tolist()) if c != -1)
    if not clusters:
        return {}
    docs = [" ".join(t for t, lab in zip(texts, labels, strict=True) if lab == c) for c in clusters]
    cv = CountVectorizer(ngram_range=(1, 2), stop_words=stop, min_df=1, token_pattern=r"(?u)\b[a-z][a-z]+\b")
    tf = cv.fit_transform(docs).toarray().astype(float)
    tf = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1)
    avg_words = tf.sum() / len(clusters)
    idf = np.log(1 + avg_words / np.maximum(tf.sum(axis=0), 1e-12))
    w = tf * idf
    vocab = np.array(cv.get_feature_names_out())
    return {c: vocab[np.argsort(-w[i])[:top_n]].tolist() for i, c in enumerate(clusters)}


def cluster_quality(y: Sequence[str], labels: np.ndarray) -> dict:
    y = np.asarray(y)
    mask = labels != -1
    return {
        "n_clusters": int(len(set(labels.tolist()) - {-1})),
        "noise_share": float(1 - mask.mean()),
        "nmi_all": float(normalized_mutual_info_score(y, labels)),
        "ari_all": float(adjusted_rand_score(y, labels)),
        "nmi_clustered": float(normalized_mutual_info_score(y[mask], labels[mask])) if mask.any() else 0.0,
        "ari_clustered": float(adjusted_rand_score(y[mask], labels[mask])) if mask.any() else 0.0,
    }


def cluster_table(texts: Sequence[str], y: Sequence[str], labels: np.ndarray, top_n: int = 6) -> list[dict]:
    """One row per cluster: size, majority intent, purity, keywords."""
    y = np.asarray(y)
    kw = ctfidf_keywords(texts, labels, top_n)
    rows = []
    for c, words in kw.items():
        m = labels == c
        vals, counts = np.unique(y[m], return_counts=True)
        j = int(np.argmax(counts))
        rows.append(
            {
                "cluster": int(c),
                "size": int(m.sum()),
                "majority_intent": str(vals[j]),
                "purity": float(counts[j] / m.sum()),
                "keywords": words,
            }
        )
    return sorted(rows, key=lambda r: -r["size"])
