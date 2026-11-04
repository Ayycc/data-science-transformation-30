"""DP synthesiser in the style of MST (McKenna, Miklau & Sheldon 2021), with a tree model instead of Private-PGM.

All accounting is in zero-concentrated DP and converted to (epsilon, delta) at the end:
1. one third of rho measures every 1-way marginal with Gaussian noise (each record touches one cell per
   marginal, so L2 sensitivity is sqrt(d) for the d marginals measured together);
2. one third selects a maximum spanning tree of attribute pairs with the exponential mechanism (d - 1
   rounds; each round is (eps_r^2 / 8)-zCDP). A pair's score is the L1 gap between its true 2-way marginal and
   the product of the noisy 1-way marginals (sensitivity 1), so pairs that are badly explained get picked;
3. one third measures the d - 1 selected 2-way marginals (L2 sensitivity sqrt(d - 1)).
Sampling is ancestral along the tree from the noisy (clipped, renormalised) conditionals. Everything after
step 3 is post-processing. Numerics use public bins (``encoding.BINS``) and are drawn uniformly inside a bin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..accounting import ZCDPAccountant, dp_to_zcdp_rho, zcdp_to_dp
from ..data import COLUMNS
from ..encoding import discretize, domain_sizes, undiscretize


def _marginal(codes: np.ndarray, cols: tuple[int, ...], sizes: list[int]) -> np.ndarray:
    shape = tuple(sizes[c] for c in cols)
    flat = np.ravel_multi_index(tuple(codes[:, c] for c in cols), shape)
    return np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape).astype(float)


@dataclass
class DPTreeSynthesizer:
    epsilon: float
    delta: float = 1e-6
    seed: int = 0
    name: str = ""
    accountant: ZCDPAccountant | None = None
    edges_: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"DP tree (MST-style) eps={self.epsilon:g}"

    def fit(self, df: pd.DataFrame) -> DPTreeSynthesizer:
        rng = np.random.default_rng(self.seed)
        codes = discretize(df)
        sizes = [domain_sizes()[c] for c in COLUMNS]
        d = len(COLUMNS)
        rho = dp_to_zcdp_rho(self.epsilon, self.delta)
        self.accountant = acc = ZCDPAccountant(rho)
        rho1 = rho2 = rho3 = rho / 3

        # 1. noisy 1-way marginals
        acc.spend("1-way marginals (Gaussian)", rho1)
        sigma1 = math.sqrt(d) / math.sqrt(2 * rho1)
        self.one_way_ = [np.clip(_marginal(codes, (j,), sizes) + rng.normal(0, sigma1, sizes[j]), 0, None) for j in range(d)]
        n_hat = float(np.mean([m.sum() for m in self.one_way_]))
        self.n_hat_ = n_hat
        probs = [m / max(m.sum(), 1e-9) for m in self.one_way_]

        # 2. exponential-mechanism Kruskal over pairs
        eps_round = math.sqrt(8 * rho2 / (d - 1))
        scores = {}
        for i in range(d):
            for j in range(i + 1, d):
                est = n_hat * np.outer(probs[i], probs[j])
                scores[(i, j)] = float(np.abs(_marginal(codes, (i, j), sizes) - est).sum())
        parent = list(range(d))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        edges: list[tuple[int, int]] = []
        for r in range(d - 1):
            cand = [p for p in scores if find(p[0]) != find(p[1])]
            s = np.array([scores[p] for p in cand])
            logits = eps_round * (s - s.max()) / 2.0  # sensitivity 1
            w = np.exp(logits)
            pick = cand[int(rng.choice(len(cand), p=w / w.sum()))]
            acc.spend(f"select pair {r + 1} (exponential mechanism)", eps_round**2 / 8)
            parent[find(pick[0])] = find(pick[1])
            edges.append(pick)
        self.edges_ = edges

        # 3. noisy 2-way marginals of the tree edges
        acc.spend("2-way marginals of tree edges (Gaussian)", rho3)
        sigma3 = math.sqrt(d - 1) / math.sqrt(2 * rho3)
        self.two_way_ = {e: np.clip(_marginal(codes, e, sizes) + rng.normal(0, sigma3, (sizes[e[0]], sizes[e[1]])), 0, None) for e in edges}
        self.sizes_ = sizes
        self.epsilon_spent_ = zcdp_to_dp(acc.spent, self.delta)
        return self

    def sample(self, n: int | None = None, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + 1 if seed is None else seed)
        n = int(round(self.n_hat_)) if n is None else n
        d = len(COLUMNS)
        adj: dict[int, list[int]] = {i: [] for i in range(d)}
        for a, b in self.edges_:
            adj[a].append(b)
            adj[b].append(a)
        root = COLUMNS.index("income")
        codes = np.zeros((n, d), dtype=np.int64)
        p = self.one_way_[root] + 1e-9
        codes[:, root] = rng.choice(len(p), size=n, p=p / p.sum())
        seen, queue = {root}, [root]
        while queue:
            par = queue.pop(0)
            for ch in adj[par]:
                if ch in seen:
                    continue
                key = (par, ch) if (par, ch) in self.two_way_ else (ch, par)
                tab = self.two_way_[key] if key == (par, ch) else self.two_way_[key].T  # rows: parent, cols: child
                fallback = self.one_way_[ch] + 1e-9
                cond = tab + 1e-9
                row_mass = tab.sum(1, keepdims=True)
                cond = np.where(row_mass > 1.0, cond, fallback[None, :])  # empty parent rows fall back to the child's marginal
                cond = cond / cond.sum(1, keepdims=True)
                cdf = cond.cumsum(1)[codes[:, par]]
                codes[:, ch] = np.minimum((cdf < rng.random(n)[:, None]).sum(1), cond.shape[1] - 1)
                seen.add(ch)
                queue.append(ch)
        return undiscretize(codes, rng)

    def structure(self) -> list[str]:
        return [f"{COLUMNS[a]} - {COLUMNS[b]}" for a, b in self.edges_]
