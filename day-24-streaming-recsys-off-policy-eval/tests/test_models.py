import numpy as np

from recsys24 import ann, metrics
from recsys24.baselines import ALS, ItemKNN, popularity_scores
from recsys24.data import temporal_split
from recsys24.two_tower import TwoTowerConfig, TwoTowerRecommender, build_sequences, recent_history


def _setup(ds):
    sp = temporal_split(ds.interactions)
    X = metrics.interaction_matrix(sp.train["user"].to_numpy(), sp.train["item"].to_numpy(), ds.n_users, ds.n_items)
    truth = metrics.interaction_matrix(sp.val["user"].to_numpy(), sp.val["item"].to_numpy(), ds.n_users, ds.n_items)
    return sp, X, truth


def test_baselines_beat_random(small_ds):
    sp, X, truth = _setup(small_ds)
    rng = np.random.default_rng(0)
    rand = metrics.ndcg_at_k(metrics.top_k(rng.random((small_ds.n_users, small_ds.n_items)), X, 10), truth, 10)
    pop = metrics.ndcg_at_k(metrics.top_k(np.broadcast_to(popularity_scores(X), X.shape), X, 10), truth, 10)
    knn = metrics.ndcg_at_k(metrics.top_k(ItemKNN(50).fit(X).score(X), X, 10), truth, 10)
    als = metrics.ndcg_at_k(metrics.top_k(ALS(16, 0.05, 5.0, 8).fit(X).score(), X, 10), truth, 10)
    assert min(pop, knn, als) > 2 * rand


def test_history_is_most_recent_first():
    import pandas as pd

    df = pd.DataFrame({"user": [0, 0, 0, 1], "item": [5, 6, 7, 2], "t": [0.1, 0.2, 0.3, 0.5]})
    seq, lens = build_sequences(df, 2)
    h = recent_history(seq, lens, 4, pad=99)
    assert h.tolist() == [[7, 6, 5, 99], [2, 99, 99, 99]]


def test_two_tower_learns_and_faiss_serves(small_ds):
    sp, X, truth = _setup(small_ds)
    cfg = TwoTowerConfig(epochs=3, batch_size=256, dim=32, hidden=64)
    tt = TwoTowerRecommender(small_ds.n_users, small_ds.n_items, small_ds.item_genres, small_ds.item_bucket, cfg).fit(sp.train)
    assert tt.history["epoch_loss"][-1] < tt.history["epoch_loss"][0]
    U, I = tt.user_embeddings(tt.histories(sp.train)), tt.item_embeddings()
    assert np.allclose(np.linalg.norm(I, axis=1), 1, atol=1e-4)
    pop = metrics.ndcg_at_k(metrics.top_k(np.broadcast_to(popularity_scores(X), X.shape), X, 10), truth, 10)
    rng = np.random.default_rng(0)
    rand = metrics.ndcg_at_k(metrics.top_k(rng.random(X.shape), X, 10), truth, 10)
    tt_ndcg = metrics.ndcg_at_k(metrics.top_k(U @ I.T, X, 10), truth, 10)
    assert tt_ndcg > 2 * rand and tt_ndcg > 0.5 * pop
    hnsw, flat = ann.build_hnsw(I), ann.build_flat(I)
    assert ann.ann_recall(hnsw, flat, U, 20) > 0.9
    seen = [X.indices[X.indptr[u] : X.indptr[u + 1]] for u in range(X.shape[0])]
    cands = ann.search_filtered(hnsw, U, seen, 20, 60)
    assert all(not np.isin(c, s).any() for c, s in zip(cands, seen))
    lat = ann.measure_latency(lambda u: tt.user_embedding_one(u, tt.histories(sp.train)[u]), hnsw, np.arange(60), k=10, warmup=10)
    assert lat["n_requests"] == 50 and lat["end_to_end"]["p99_ms"] >= lat["end_to_end"]["p50_ms"]
