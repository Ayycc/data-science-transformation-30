import numpy as np
import pytest
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from ticket_intel.active import labels_to_match, schedule, simulate
from ticket_intel.features import make_tfidf
from ticket_intel.pipeline import abstention_threshold
from ticket_intel.text import normalise
from ticket_intel.topics import cluster_quality, ctfidf_keywords


@pytest.fixture(scope="module")
def feats(tickets):
    norm = [normalise(t) for t in tickets.text]
    tf = make_tfidf().fit(norm)
    X = tf.transform(norm)
    y = tickets.intent.to_numpy()
    Z = normalize(TruncatedSVD(30, random_state=0).fit_transform(X))
    return X, y, Z


@pytest.mark.parametrize("strategy", ["random", "margin", "entropy", "margin_diverse"])
def test_al_respects_budget(feats, strategy):
    X, y, Z = feats
    labels = sorted(set(y))
    init, batch, budget = schedule(len(y), 0.05, 0.03, 0.2)
    run = simulate(X, y, X, y, labels, "tfidf", strategy, init, batch, budget, seed=1, Z_pool=Z)
    assert run.n_labels[0] == init and run.n_labels[-1] == budget
    assert len(set(run.labelled_idx.tolist())) == budget
    assert all(0 <= f <= 1 for f in run.macro_f1)


def test_al_seed_set_shared_across_strategies(feats):
    X, y, Z = feats
    labels = sorted(set(y))
    a = simulate(X, y, X, y, labels, "tfidf", "random", 30, 10, 30, seed=3, Z_pool=Z)
    b = simulate(X, y, X, y, labels, "tfidf", "margin", 30, 10, 30, seed=3, Z_pool=Z)
    assert np.array_equal(a.labelled_idx, b.labelled_idx)


def test_labels_to_match():
    assert labels_to_match([0.1, 0.5, 0.9], [10, 20, 30], 0.5) == 20
    assert labels_to_match([0.1, 0.2], [10, 20], 0.5) is None


def test_abstention_threshold():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 0], dtype=float)
    assert abstention_threshold(conf, correct, 0.95) == 0.8


def test_ctfidf_and_quality():
    texts = ["refund money back", "refund my money", "password reset link", "reset password now"]
    lab = np.array([0, 0, 1, 1])
    kw = ctfidf_keywords(texts, lab, top_n=2)
    assert "refund" in kw[0] or "money" in kw[0]
    assert "password" in kw[1] or "reset" in kw[1]
    q = cluster_quality(["a", "a", "b", "b"], lab)
    assert q["nmi_all"] == pytest.approx(1.0) and q["n_clusters"] == 2
