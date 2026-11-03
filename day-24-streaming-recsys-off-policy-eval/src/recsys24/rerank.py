"""Gradient-boosted re-ranker (LightGBM LambdaRank) over two-tower candidates, plus MMR diversity re-ranking.

Features are computed for any (user, candidate list) pair, so the same code scores retrieval candidates for the
offline evaluation and arbitrary homepage slates for the off-policy evaluation. ALS is deliberately *not* a feature,
so "two-tower + re-ranker vs ALS" is a clean comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp

FEATURES = [
    "tt_score",
    "log_pop",
    "log_recent_pop",
    "item_age",
    "genre_affinity",
    "knn_score",
    "log_hist_len",
    "user_recent_pop_taste",
    "log_pop_at_user_time",
    "recent_sim",
    "knn_recent",
]


@dataclass
class FeatureContext:
    """Everything the re-ranker needs, computed from one fit window (train, or train+val)."""

    user_emb: np.ndarray
    item_emb: np.ndarray
    knn_scores: np.ndarray  # dense users x items
    item_pop: np.ndarray
    item_recent_pop: np.ndarray
    item_first_seen: np.ndarray
    user_last_t: np.ndarray
    user_genre: np.ndarray  # users x genres (normalised profile of the last 20 titles)
    item_genre: np.ndarray  # items x genres (L2-normalised)
    user_hist_len: np.ndarray
    user_pop_taste: np.ndarray  # mean log-popularity of the user's last 20 titles
    item_time_pop: np.ndarray  # n_time_bins x items: watch counts per time bin
    user_time_bin: np.ndarray  # bin of each user's latest fit event
    user_recent_items: np.ndarray  # users x 5 latest titles (-1 padded)
    knn_recent_scores: np.ndarray  # dense users x items: item-kNN score from the last 10 titles only


def build_context(
    fit_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    item_genres: np.ndarray,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    knn_scores: np.ndarray,
    knn_sim: sp.csr_matrix,
    recent_window: float = 0.1,
    n_time_bins: int = 40,
) -> FeatureContext:
    df = fit_df.sort_values(["user", "t"], kind="stable")
    pop = np.bincount(df["item"], minlength=n_items).astype(np.float32)
    t_max = float(df["t"].max())
    rec = df[df["t"] >= t_max - recent_window]
    recent_pop = np.bincount(rec["item"], minlength=n_items).astype(np.float32)
    first_seen = np.full(n_items, t_max, dtype=np.float32)
    fs = df.groupby("item")["t"].min()
    first_seen[fs.index.to_numpy()] = fs.to_numpy()
    last_t = np.zeros(n_users, dtype=np.float32)
    lt = df.groupby("user")["t"].max()
    last_t[lt.index.to_numpy()] = lt.to_numpy()
    last20 = df.groupby("user").tail(20)
    g = item_genres / np.linalg.norm(item_genres, axis=1, keepdims=True).clip(1e-9)
    ug = np.zeros((n_users, item_genres.shape[1]), dtype=np.float32)
    np.add.at(ug, last20["user"].to_numpy(), g[last20["item"].to_numpy()])
    ug /= np.linalg.norm(ug, axis=1, keepdims=True).clip(1e-9)
    hist_len = np.bincount(df["user"], minlength=n_users).astype(np.float32)
    lp = np.log1p(pop)
    taste = np.zeros(n_users, dtype=np.float32)
    np.add.at(taste, last20["user"].to_numpy(), lp[last20["item"].to_numpy()])
    taste /= np.bincount(last20["user"], minlength=n_users).clip(1)
    t_min = float(df["t"].min())
    bins = np.clip(((df["t"].to_numpy() - t_min) / (t_max - t_min + 1e-9) * n_time_bins).astype(int), 0, n_time_bins - 1)
    time_pop = np.zeros((n_time_bins, n_items), dtype=np.float32)
    np.add.at(time_pop, (bins, df["item"].to_numpy()), 1.0)
    user_bin = np.clip(((last_t - t_min) / (t_max - t_min + 1e-9) * n_time_bins).astype(int), 0, n_time_bins - 1)
    last5 = df.groupby("user").tail(5)
    pos = last5.groupby("user").cumcount().to_numpy()
    recent_items = np.full((n_users, 5), -1, dtype=np.int64)
    recent_items[last5["user"].to_numpy(), pos] = last5["item"].to_numpy()
    last10 = df.groupby("user").tail(10)
    xr = sp.csr_matrix((np.ones(len(last10), dtype=np.float32), (last10["user"].to_numpy(), last10["item"].to_numpy())), shape=(n_users, n_items))
    knn_recent = np.asarray((xr @ knn_sim).todense(), dtype=np.float32)
    return FeatureContext(
        user_emb,
        item_emb,
        knn_scores,
        pop,
        recent_pop,
        first_seen,
        last_t,
        ug,
        g.astype(np.float32),
        hist_len,
        taste,
        time_pop,
        user_bin,
        recent_items,
        knn_recent,
    )


def features(ctx: FeatureContext, users: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """Feature tensor (n_users x K x F) for candidate lists ``cands`` (n_users x K)."""
    u = users[:, None]
    tt = np.einsum("nd,nkd->nk", ctx.user_emb[users], ctx.item_emb[cands])
    lp = np.log1p(ctx.item_pop[cands])
    lrp = np.log1p(ctx.item_recent_pop[cands])
    age = ctx.user_last_t[u] - ctx.item_first_seen[cands]
    ga = np.einsum("ng,nkg->nk", ctx.user_genre[users], ctx.item_genre[cands])
    knn = ctx.knn_scores[u, cands]
    hl = np.broadcast_to(np.log1p(ctx.user_hist_len[u]), cands.shape)
    pt = np.broadcast_to(ctx.user_pop_taste[u], cands.shape) - lp
    b = ctx.user_time_bin[u]
    tpop = np.log1p(ctx.item_time_pop[b, cands] + ctx.item_time_pop[np.clip(b - 1, 0, None), cands])
    ri = ctx.user_recent_items[users]  # n x 5
    re = np.where((ri >= 0)[..., None], ctx.item_emb[np.clip(ri, 0, None)], 0.0)  # n x 5 x d
    rsim = np.einsum("nkd,nrd->nkr", ctx.item_emb[cands], re).sum(2) / np.maximum((ri >= 0).sum(1), 1)[:, None]
    kr = ctx.knn_recent_scores[u, cands]
    return np.stack([tt, lp, lrp, age, ga, knn, hl, pt, tpop, rsim, kr], axis=-1).astype(np.float32)


def labels_for(cands: np.ndarray, truth: sp.csr_matrix, users: np.ndarray) -> np.ndarray:
    return np.asarray(truth[users[:, None], cands].todense() > 0).astype(np.int32)


class ReRanker:
    """LightGBM re-ranker; ``objective`` is "binary" (pointwise log-loss) or "lambdarank" (listwise NDCG).

    Labels are sparse (5 held-out titles per user), so the model is kept small and strongly regularised.
    """

    def __init__(self, objective: str = "binary", feature_names: list[str] | None = None, n_estimators: int = 100, seed: int = 11):
        self.objective = objective
        self.feature_names = list(feature_names or FEATURES)
        params = dict(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=200,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=4,
            verbose=-1,
        )
        self.model = (
            lgb.LGBMRanker(objective="lambdarank", **params) if objective == "lambdarank" else lgb.LGBMClassifier(objective="binary", **params)
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> ReRanker:
        """``X``: n x K x F, ``y``: n x K. Lists without a positive carry no ranking signal and are dropped."""
        keep = y.sum(1) > 0
        X, y = X[keep], y[keep]
        n, k, f = X.shape
        if self.objective == "lambdarank":
            self.model.fit(X.reshape(n * k, f), y.reshape(-1), group=np.full(n, k), feature_name=self.feature_names)
        else:
            self.model.fit(X.reshape(n * k, f), y.reshape(-1), feature_name=self.feature_names)
        self.n_train_lists = int(n)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        n, k, f = X.shape
        flat = X.reshape(n * k, f)
        s = self.model.predict(flat) if self.objective == "lambdarank" else self.model.predict_proba(flat)[:, 1]
        return s.reshape(n, k)

    def importance(self) -> dict:
        gain = self.model.booster_.feature_importance(importance_type="gain")
        return {name: float(v / gain.sum()) for name, v in zip(self.feature_names, gain, strict=True)}


def rerank(cands: np.ndarray, scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(cands, order, 1)


def mmr(cands: np.ndarray, scores: np.ndarray, item_emb: np.ndarray, k: int = 10, lam: float = 0.7) -> np.ndarray:
    """Maximal marginal relevance: greedily pick lam * relevance - (1 - lam) * max cosine to already-picked items."""
    n, m = cands.shape
    rel = (scores - scores.min(1, keepdims=True)) / np.ptp(scores, axis=1, keepdims=True).clip(1e-9)
    emb = item_emb[cands]  # n x m x d (already L2-normalised)
    sim = np.einsum("nmd,nld->nml", emb, emb)
    chosen = np.zeros((n, k), dtype=np.int64)
    picked = np.zeros((n, m), dtype=bool)
    max_sim = np.zeros((n, m))
    rows = np.arange(n)
    for j in range(k):
        obj = lam * rel - (1 - lam) * max_sim
        obj[picked] = -np.inf
        c = obj.argmax(1)
        chosen[:, j] = c
        picked[rows, c] = True
        max_sim = np.maximum(max_sim, sim[rows, c])
    return np.take_along_axis(cands, chosen, 1)
