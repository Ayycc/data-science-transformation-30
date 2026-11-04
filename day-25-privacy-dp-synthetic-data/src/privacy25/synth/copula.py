"""Gaussian copula synthesiser (not DP), in the style of SDV's ``GaussianCopulaSynthesizer``.

Each column is mapped to a standard normal through its empirical CDF: numerics by mid-rank (ties broken at
random, which handles the spike at capital_gain = 0), categoricals by giving every level an interval of
[0, 1] as wide as its frequency and drawing uniformly inside it (SDV's frequency encoding). A full
correlation matrix is fitted on the normal scores; sampling inverts each marginal (numerics via the
empirical quantile function, so values stay inside the observed range).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..data import CATEGORICAL, COLUMNS, NUMERIC_BOUNDS


@dataclass
class GaussianCopula:
    seed: int = 0
    name: str = "Gaussian copula"

    def fit(self, df: pd.DataFrame) -> GaussianCopula:
        rng = np.random.default_rng(self.seed)
        n = len(df)
        Z = np.empty((n, len(COLUMNS)))
        self.numeric_sorted_: dict[str, np.ndarray] = {}
        self.cat_edges_: dict[str, tuple[list[str], np.ndarray]] = {}
        for j, c in enumerate(COLUMNS):
            if c in NUMERIC_BOUNDS:
                v = df[c].to_numpy(dtype=float)
                r = stats.rankdata(v + rng.random(n) * 1e-6)  # random tie-breaking
                u = (r - 0.5) / n
                self.numeric_sorted_[c] = np.sort(v)
            else:
                levels = CATEGORICAL[c]
                freq = df[c].value_counts().reindex(levels, fill_value=0).to_numpy() / n
                edges = np.concatenate([[0.0], np.cumsum(freq)])
                code = pd.Categorical(df[c], categories=levels).codes
                u = edges[code] + rng.random(n) * freq[code]
                self.cat_edges_[c] = (levels, edges)
            Z[:, j] = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        self.corr_ = np.corrcoef(Z, rowvar=False)
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + 1 if seed is None else seed)
        Z = rng.multivariate_normal(np.zeros(len(COLUMNS)), self.corr_, size=n, method="eigh")
        U = stats.norm.cdf(Z)
        cols = {}
        for j, c in enumerate(COLUMNS):
            if c in NUMERIC_BOUNDS:
                srt = self.numeric_sorted_[c]
                cols[c] = np.round(np.quantile(srt, U[:, j], method="linear")).astype(int)
            else:
                levels, edges = self.cat_edges_[c]
                k = np.clip(np.searchsorted(edges, U[:, j], side="right") - 1, 0, len(levels) - 1)
                cols[c] = np.asarray(levels, dtype=object)[k]
        return pd.DataFrame(cols)[COLUMNS]
