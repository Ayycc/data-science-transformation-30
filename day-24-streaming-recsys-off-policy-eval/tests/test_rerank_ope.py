import numpy as np
import pytest

from recsys24 import ope
from recsys24.rerank import mmr, rerank


def test_rerank_and_mmr():
    cands = np.array([[10, 11, 12, 13]])
    scores = np.array([[0.1, 0.9, 0.5, 0.8]])
    assert rerank(cands, scores).tolist() == [[11, 13, 12, 10]]
    emb = np.zeros((14, 2))
    emb[10:] = [[1, 0], [1, 0], [0, 1], [1, 0]]  # 11 and 13 are duplicates
    assert mmr(cands, scores, emb, k=2, lam=1.0).tolist() == [[11, 13]]
    assert mmr(cands, scores, emb, k=2, lam=0.5).tolist() == [[11, 12]]


def _toy(n_users=400, k=6, seed=0):
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0.02, 0.4, size=(n_users, k))
    pop = rng.normal(size=(n_users, k))
    cfg = ope.OPEConfig(slate_size=k)
    pi0 = ope.logging_policy(pop, cfg)
    pols = ope.target_policies({"good": np.log(p_true) + rng.normal(0, 0.3, size=p_true.shape)}, cfg, greedy=("good",))
    X = np.stack([np.log(p_true) + rng.normal(0, 0.5, size=p_true.shape), pop], -1)
    return p_true, pi0, pols, X, cfg


def test_logging_policy_is_valid():
    p_true, pi0, pols, _, cfg = _toy()
    assert np.allclose(pi0.sum(1), 1) and pi0.min() >= cfg.epsilon / cfg.slate_size - 1e-12
    for pi in pols.values():
        assert np.allclose(pi.sum(1), 1)
    tv = ope.true_values(pols, p_true)
    assert tv["good (greedy)"] >= tv["good"] - 0.05 and tv["good"] > tv["Uniform random"]


def test_estimators_are_close_to_truth_on_average():
    p_true, pi0, pols, X, _ = _toy()
    truth = ope.true_values(pols, p_true)
    res = ope.replicate(3000, 6, pi0, p_true, pols, X, seed=1)
    s = ope.summarise(res, truth)
    for name in pols:
        for e in ("IPS", "SNIPS", "DR"):
            assert abs(s[name][e]["bias"]) < 4 * s[name][e]["std"] / np.sqrt(6) + 0.01
        assert s[name]["DR"]["rmse"] < 0.05


def test_ips_is_exactly_unbiased_in_expectation():
    p_true, pi0, pols, _, _ = _toy(n_users=50, k=4)
    pi = pols["good"]
    # E[w r] over a ~ pi0, r ~ p = sum_a pi0 * (pi/pi0) * p = sum_a pi * p
    assert np.mean((pi0 * (pi / pi0) * p_true).sum(1)) == pytest.approx(ope.true_values({"g": pi}, p_true)["g"])
