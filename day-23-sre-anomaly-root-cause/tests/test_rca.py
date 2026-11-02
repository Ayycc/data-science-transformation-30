import numpy as np

from sreanomaly import rca
from sreanomaly.pipeline import loo_likelihood_ratio
from sreanomaly.topology import SERVICES, call_graph


def test_onset_pelt_and_binseg_find_step():
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(0, 0.3, 40), rng.normal(4, 0.3, 20)]
    for m in ("pelt", "binseg"):
        assert abs(rca.onset(x, m) - 40) <= 1
    assert rca.onset(rng.normal(0, 0.3, 60), "pelt") is None


def _chain_world(seed=0):
    """order-db fault at t=60 propagating to order (lag 1) and checkout (lag 2)."""
    rng = np.random.default_rng(seed)
    T, S = 120, len(SERVICES)
    sig = np.abs(rng.normal(0, 0.3, (T, S)))
    root_drive = np.r_[np.zeros(60), np.linspace(1, 5, 60)] + rng.normal(0, 0.3, T)
    chain = {"order-db": 0, "order": 1, "checkout": 2, "api-gateway": 3}
    for s, lag in chain.items():
        j = SERVICES.index(s)
        sig[lag:, j] += (0.8**lag) * root_drive[: T - lag]
    mv = sig / 1.0
    return sig, mv, list(chain)


def test_pc_respects_call_graph_direction():
    sig, _, svcs = _chain_world()
    g = call_graph()
    X = sig[40:100, [SERVICES.index(s) for s in svcs]]
    edges = rca._pc_edges(X, svcs, g, 0.05)
    for cause, effect in edges:
        assert g.has_edge(effect, cause)  # effect calls cause


def test_rank_incident_finds_chain_root():
    sig, mv, svcs = _chain_world()
    res = rca.rank_incident(sig, mv, SERVICES, call_graph(), svcs[1:], first_alert=62)
    assert "order-db" in res.candidates  # added as a callee of an alerting service
    for m in ("topology_frontier", "earliest_onset_pelt", "causal_pagerank", "ensemble"):
        assert rca.topk_hit(res.scores[m], "order-db", 1), m


def test_deploy_prior_breaks_tie_and_loo_ratio():
    s = {"a": 1.0, "b": 1.0}
    assert rca.topk_hit(rca.apply_deploy_prior(s, {"b"}, 5.0), "b", 1)
    rows = [dict(incident_id=i, root="a", candidates=["a", "b", "c"], deployed={"a"} if i < 3 else set()) for i in range(4)]
    lr = loo_likelihood_ratio(rows, exclude=0)
    # roots: 2 of 3 deployed -> (2+1)/(3+2)=0.6 ; others 0 of 6 -> 1/8 ; ratio 4.8
    assert abs(lr - 4.8) < 1e-9
