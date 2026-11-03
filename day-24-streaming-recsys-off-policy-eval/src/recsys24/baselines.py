"""Baselines: popularity, item-kNN (implicit CosineRecommender similarity) and ALS (implicit)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from implicit.cpu.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender
from threadpoolctl import threadpool_limits


def popularity_scores(user_items: sp.csr_matrix) -> np.ndarray:
    return np.asarray(user_items.sum(0)).ravel().astype(np.float32)


class ItemKNN:
    """Item-item cosine neighbours (top ``k`` per item) from ``implicit``; user score = sum of similarities to history."""

    def __init__(self, k: int = 100):
        self.k = k
        self.sim: sp.csr_matrix | None = None

    def fit(self, user_items: sp.csr_matrix) -> ItemKNN:
        m = CosineRecommender(K=self.k, num_threads=4)
        with threadpool_limits(1, "blas"):
            m.fit(user_items.tocsr(), show_progress=False)
        self.sim = m.similarity.tocsr().astype(np.float32)
        self.sim.setdiag(0)
        self.sim.eliminate_zeros()
        return self

    def score(self, user_items: sp.csr_matrix) -> np.ndarray:
        return np.asarray((user_items @ self.sim).todense(), dtype=np.float32)


class ALS:
    """Weighted matrix factorisation for implicit feedback (Hu, Koren & Volinsky 2008) via ``implicit``."""

    def __init__(self, factors: int = 64, regularization: float = 0.05, alpha: float = 10.0, iterations: int = 15, seed: int = 0):
        self.params = {"factors": factors, "regularization": regularization, "alpha": alpha, "iterations": iterations}
        self.seed = seed
        self.model: AlternatingLeastSquares | None = None

    def fit(self, user_items: sp.csr_matrix) -> ALS:
        self.model = AlternatingLeastSquares(**self.params, random_state=self.seed, num_threads=4)
        with threadpool_limits(1, "blas"):  # implicit parallelises with OpenMP; nested BLAS threads oversubscribe
            self.model.fit(user_items.tocsr(), show_progress=False)
        return self

    @property
    def user_factors(self) -> np.ndarray:
        return np.asarray(self.model.user_factors)

    @property
    def item_factors(self) -> np.ndarray:
        return np.asarray(self.model.item_factors)

    def score(self) -> np.ndarray:
        return (self.user_factors @ self.item_factors.T).astype(np.float32)
