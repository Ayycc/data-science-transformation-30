"""Pool-based active learning with a simulated annotator (the oracle reveals the gold intent)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans

from .models import full_proba, macro_f1, make_clf

warnings.filterwarnings("ignore", message="The number of unique classes")

STRATEGIES = ("random", "margin", "entropy", "margin_diverse", "cluster_margin")


@dataclass
class ALRun:
    strategy: str
    kind: str
    repeat: int
    n_labels: list[int] = field(default_factory=list)
    macro_f1: list[float] = field(default_factory=list)
    labelled_idx: np.ndarray | None = None


def _diverse_pick(Z: np.ndarray, order: np.ndarray, k: int, seed: int, pool_factor: int = 10) -> np.ndarray:
    """Take the `pool_factor*k` most uncertain items, k-means them into k groups, keep the most uncertain per group."""
    top = order[: min(len(order), pool_factor * k)]
    if len(top) <= k:
        return top
    km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(Z[top])
    picked = []
    for c in range(k):
        members = np.flatnonzero(km.labels_ == c)
        if len(members):
            picked.append(top[members.min()])  # `top` is sorted by uncertainty, so min position = most uncertain
    rest = [i for i in top if i not in set(picked)]
    return np.array(picked + rest[: k - len(picked)])


def cluster_seed(Z: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Cold start without labels: the item closest to each of k k-means centroids."""
    km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(Z)
    d = km.transform(Z)
    out = []
    for c in range(k):
        members = np.flatnonzero(km.labels_ == c)
        if len(members):
            out.append(members[np.argmin(d[members, c])])
    return np.array(sorted(set(out)))


def _query(strategy: str, proba: np.ndarray, k: int, rng: np.random.Generator, Z: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Return positions (into the unlabelled candidates) of the k most informative items."""
    n = proba.shape[0]
    k = min(k, n)
    if strategy == "random":
        return rng.choice(n, size=k, replace=False)
    if strategy in ("margin_diverse", "cluster_margin"):
        top2 = np.sort(proba, axis=1)[:, -2:]
        order = np.argsort(top2[:, 1] - top2[:, 0], kind="stable")
        return _diverse_pick(Z, order, k, seed)
    if strategy == "margin":
        top2 = np.sort(proba, axis=1)[:, -2:]
        score = top2[:, 1] - top2[:, 0]  # small margin = uncertain
        return np.argsort(score, kind="stable")[:k]
    if strategy == "entropy":
        p = np.clip(proba, 1e-12, 1.0)
        score = -(p * np.log(p)).sum(axis=1)
        return np.argsort(-score, kind="stable")[:k]
    raise ValueError(strategy)


def schedule(pool_size: int, init_frac: float, batch_frac: float, budget_frac: float) -> tuple[int, int, int]:
    budget = max(int(round(pool_size * budget_frac)), 2)
    init = max(int(round(pool_size * init_frac)), 2)
    batch = max(int(round(pool_size * batch_frac)), 1)
    return init, batch, budget


def simulate(
    X_pool,
    y_pool: np.ndarray,
    X_test,
    y_test: np.ndarray,
    labels: Sequence[str],
    kind: str,
    strategy: str,
    init_n: int,
    batch_n: int,
    budget_n: int,
    seed: int,
    repeat: int = 0,
    Z_pool: np.ndarray | None = None,
) -> ALRun:
    """Start from a seed set, then query `batch_n` labels per round until `budget_n`.

    The seed set is random (identical for every strategy within a repeat), except for
    `cluster_margin`, which picks k-means medoids of `Z_pool` (a dense, label-free projection).
    """
    rng = np.random.default_rng(seed + 1000 * repeat)
    n = len(y_pool)
    labelled = np.zeros(n, dtype=bool)
    random_seed_set = rng.choice(n, size=init_n, replace=False)
    if strategy == "cluster_margin":
        labelled[cluster_seed(Z_pool, init_n, seed + repeat)] = True
    else:
        labelled[random_seed_set] = True
    q_rng = np.random.default_rng(seed + 1000 * repeat + 7)
    run = ALRun(strategy=strategy, kind=kind, repeat=repeat)
    while True:
        idx = np.flatnonzero(labelled)
        clf = make_clf(kind, seed).fit(X_pool[idx], y_pool[idx])
        run.n_labels.append(int(labelled.sum()))
        run.macro_f1.append(macro_f1(y_test, clf.predict(X_test), labels))
        if labelled.sum() >= budget_n:
            break
        cand = np.flatnonzero(~labelled)
        k = min(batch_n, budget_n - int(labelled.sum()))
        proba = full_proba(clf, X_pool[cand], labels) if strategy != "random" else np.empty((len(cand), 0))
        Z = Z_pool[cand] if Z_pool is not None else None
        labelled[cand[_query(strategy, proba, k, q_rng, Z, seed + repeat)]] = True
    run.labelled_idx = np.flatnonzero(labelled)
    return run


def labels_to_match(al_curve: Sequence[float], al_n: Sequence[int], target: float) -> int | None:
    """Smallest label count at which the AL curve first reaches `target` (None if never)."""
    for n, f in zip(al_n, al_curve, strict=True):
        if f >= target:
            return int(n)
    return None
