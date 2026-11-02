"""Root-cause ranking for one grouped incident.

Signals: change-point onsets (ruptures PELT / binary segmentation), PC causal discovery (causal-learn) with
call-graph background knowledge, pairwise Granger tests (statsmodels) along call edges, and a personalised
PageRank walk from symptoms to causes (NetworkX). A deploy-event prior is applied on top of any ranking."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import ruptures as rpt

PRE, POST = 45, 15  # analysis window: [first alert - PRE, min(last alert, first alert + SPAN_CAP) + POST] (minutes)
SPAN_CAP = 60
ONSET_DELTA = 1.5  # z-units above the window's first segment that counts as an anomalous onset


def anomaly_signal(z: np.ndarray) -> np.ndarray:
    """Per-minute, per-service anomaly signal: mean of the upward (positive) z-scores across metrics. (T, S)."""
    return np.clip(z, 0, None).mean(axis=2)


def onset(series: np.ndarray, method: str = "pelt", pen: float | None = None) -> int | None:
    """Index of the first change point after which the signal's level rises by >= ONSET_DELTA."""
    x = series.astype(float)
    n = len(x)
    if n < 8:
        return None
    sd = max(np.median(np.abs(np.diff(x))) / 0.954, 0.15)  # robust noise scale from first differences
    pen = pen if pen is not None else 2.0 * np.log(n) * sd**2 * 4
    algo = rpt.Pelt(model="l2", min_size=2, jump=1) if method == "pelt" else rpt.Binseg(model="l2", min_size=2, jump=1)
    bkps = algo.fit(x.reshape(-1, 1)).predict(pen=pen)
    edges = [0] + bkps
    base = x[: edges[1]].mean()
    for a, b in zip(edges[:-1], edges[1:]):
        if a > 0 and x[a:b].mean() - base >= ONSET_DELTA:
            return a
    return None


@dataclass
class RCAResult:
    candidates: list[str]
    scores: dict[str, dict[str, float]] = field(default_factory=dict)  # method -> service -> score (higher = more likely root)
    onsets: dict[str, dict[str, int | None]] = field(default_factory=dict)  # cp method -> service -> absolute minute
    causal_edges: list[tuple[str, str, float]] = field(default_factory=list)  # (cause, effect, weight)


def _pc_edges(X: np.ndarray, names: list[str], g: nx.DiGraph, alpha: float) -> set[tuple[str, str]]:
    """PC (Fisher-z) with background knowledge: an edge may exist only between call-graph neighbours and only
    in the callee -> caller direction (faults flow against calls). Returns oriented (cause, effect) edges."""
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    nodes = [GraphNode(f"X{i + 1}") for i in range(len(names))]
    bk = BackgroundKnowledge()
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j and not g.has_edge(b, a):  # allow a -> b only if b calls a
                bk.add_forbidden_by_node(nodes[i], nodes[j])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cg = pc(X, alpha=alpha, indep_test="fisherz", background_knowledge=bk, show_progress=False, node_names=[n.get_name() for n in nodes])
    G = cg.G.graph
    out = set()
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j or G[i, j] == 0:
                continue
            # causal-learn: G[j,i]==1 and G[i,j]==-1 means i -> j; undirected is -1/-1
            if (G[j, i] == 1 and G[i, j] == -1) or (G[i, j] == -1 and G[j, i] == -1 and g.has_edge(names[j], names[i])):
                out.add((names[i], names[j]))
    return out


def _granger_p(cause: np.ndarray, effect: np.ndarray, maxlag: int = 3) -> float:
    from statsmodels.tsa.stattools import grangercausalitytests

    data = np.column_stack([effect, cause])
    if np.std(cause) < 1e-6 or np.std(effect) < 1e-6:
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
        except Exception:
            return 1.0
    return float(min(r[0]["ssr_ftest"][1] for r in res.values()))


def _norm(d: dict[str, float]) -> dict[str, float]:
    tot = sum(max(v, 0) for v in d.values())
    return {k: (max(v, 0) / tot if tot > 0 else 1 / len(d)) for k, v in d.items()}


def rank_incident(
    sig: np.ndarray,
    mv_scores: np.ndarray,
    services: list[str],
    g: nx.DiGraph,
    group_services: list[str],
    first_alert: int,
    last_alert: int | None = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> RCAResult:
    """Score root-cause candidates for one grouped incident with several methods (no deploy prior here)."""
    idx = {s: i for i, s in enumerate(services)}
    last = first_alert if last_alert is None else min(last_alert, first_alert + SPAN_CAP)
    lo, hi = max(0, first_alert - PRE), min(sig.shape[0], last + POST)
    cand = sorted(set(group_services) | {c for s in group_services for c in g.successors(s)})
    res = RCAResult(candidates=cand)
    W = sig[lo:hi, [idx[c] for c in cand]]
    peak = {c: float(mv_scores[lo:hi, idx[c]].max()) for c in cand}
    anomalous = {c for c in cand if peak[c] > 1.0}

    # 1) most anomalous service
    res.scores["max_anomaly"] = peak

    # 2) change-point onsets (PELT and binary segmentation): earliest anomalous onset wins
    for m in ("pelt", "binseg"):
        ons = {}
        for k, c in enumerate(cand):
            o = onset(W[:, k], m)
            ons[c] = None if o is None else lo + o
        res.onsets[m] = ons
        valid = [v for v in ons.values() if v is not None]
        t_min = min(valid) if valid else first_alert
        res.scores[f"earliest_onset_{m}"] = {
            c: (np.exp(-(ons[c] - t_min) / 3.0) * (1 + 0.01 * min(peak[c], 5)) if ons[c] is not None else 1e-4 * peak[c]) for c in cand
        }

    # 3) topology frontier: anomalous services none of whose callees are anomalous
    res.scores["topology_frontier"] = {c: peak[c] * (0.1 if any(s in anomalous for s in g.successors(c)) else 1.0) for c in cand}

    # 4) causal graph: PC with call-graph background knowledge + Granger weights -> personalised PageRank
    rng = np.random.default_rng(seed)
    keep = [k for k in range(len(cand)) if np.std(W[:, k]) > 1e-3]
    names = [cand[k] for k in keep]
    Xc = W[:, keep] + rng.normal(0, 1e-3, (W.shape[0], len(keep)))
    pc_set: set[tuple[str, str]] = set()
    if len(names) >= 2:
        try:
            pc_set = _pc_edges(Xc, names, g, alpha)
        except Exception:
            pc_set = set()
    H = nx.DiGraph()
    H.add_nodes_from(cand)
    for caller in cand:
        for callee in g.successors(caller):
            if callee not in cand:
                continue
            p = _granger_p(W[:, cand.index(callee)], W[:, cand.index(caller)])
            w = (1.0 if (callee, caller) in pc_set else 0.25) * (1 - p) + 0.05
            res.causal_edges.append((callee, caller, float(w)))
            H.add_edge(caller, callee, weight=w)  # walk from effect to cause
    pers = _norm({c: min(peak[c], 5.0) for c in cand})
    pr = nx.pagerank(H, alpha=0.85, personalization=pers, weight="weight") if H.number_of_edges() else pers
    res.scores["causal_pagerank"] = {c: pr[c] * min(peak[c], 5.0) for c in cand}

    # 5) ensemble: mean of normalised onset (PELT), frontier and causal scores
    parts = [_norm(res.scores[k]) for k in ("earliest_onset_pelt", "topology_frontier", "causal_pagerank")]
    res.scores["ensemble"] = {c: float(np.mean([p[c] for p in parts])) for c in cand}
    return res


def apply_deploy_prior(scores: dict[str, float], deployed: set[str], lr: float) -> dict[str, float]:
    """Posterior ∝ normalised method score x prior, prior = lr for services with a recent deploy, else 1."""
    p = _norm(scores)
    return {c: p[c] * (lr if c in deployed else 1.0) for c in p}


def topk_hit(scores: dict[str, float], root: str, k: int) -> bool:
    order = sorted(scores, key=lambda c: (-scores[c], c))
    return root in order[:k]
