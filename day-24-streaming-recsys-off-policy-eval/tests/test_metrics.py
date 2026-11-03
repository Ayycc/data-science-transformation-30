import numpy as np
import pytest

from recsys24 import metrics


def test_top_k_excludes_seen_and_orders():
    scores = np.array([[0.9, 0.8, 0.1, 0.5], [0.1, 0.2, 0.3, 0.4]])
    seen = metrics.interaction_matrix(np.array([0]), np.array([0]), 2, 4)
    recs = metrics.top_k(scores, seen, 2)
    assert recs.tolist() == [[1, 3], [3, 2]]


def test_recall_and_ndcg_hand_computed():
    truth = metrics.interaction_matrix(np.array([0, 0, 1]), np.array([1, 3, 2]), 2, 5)
    recs = np.array([[1, 0, 3], [4, 0, 1]])
    assert metrics.recall_at_k(recs, truth, 3) == pytest.approx((1.0 + 0.0) / 2)
    dcg = 1.0 + 1 / np.log2(4)
    idcg = 1.0 + 1 / np.log2(3)
    assert metrics.ndcg_at_k(recs, truth, 3) == pytest.approx((dcg / idcg + 0.0) / 2)


def test_coverage_novelty_diversity():
    recs = np.array([[0, 1], [0, 2]])
    assert metrics.coverage(recs, 4) == 0.75
    pop = np.array([9.0, 1.0, 1.0, 0.0])
    assert metrics.novelty(recs, pop, 9) > -np.log2(10 / 10)
    genres = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    assert metrics.intra_list_diversity(np.array([[0, 1]]), genres) == pytest.approx(0.0)
    assert metrics.intra_list_diversity(np.array([[0, 2]]), genres) == pytest.approx(1.0)
    prof = metrics.popularity_profile(recs, np.array([10.0, 5, 1, 0]), head_frac=0.25)
    assert prof["head_share"] == 0.5
