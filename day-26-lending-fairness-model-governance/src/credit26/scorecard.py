"""Baseline WOE scorecard: quantile binning, weight of evidence, logistic regression, points scaling, points-below-max reasons."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class WOEBinner:
    """Per-feature bins (quantile edges for numerics, levels for categoricals) and their WOE = ln(%good / %bad), smoothed."""

    max_bins: int = 6
    min_share: float = 0.05
    smoothing: float = 0.5
    edges: dict[str, np.ndarray] = field(default_factory=dict)
    woe: dict[str, dict] = field(default_factory=dict)
    iv: dict[str, float] = field(default_factory=dict)

    def _bin(self, col: str, x: pd.Series) -> np.ndarray:
        if col in self.edges:
            return np.searchsorted(self.edges[col], x.to_numpy(dtype=float), side="right")
        return x.astype(str).to_numpy()

    def fit(self, X: pd.DataFrame, y: np.ndarray, numeric: list[str], categorical: list[str]) -> WOEBinner:
        y = np.asarray(y)
        for c in numeric:
            qs = np.unique(np.quantile(X[c].to_numpy(dtype=float), np.linspace(0, 1, self.max_bins + 1)[1:-1]))
            self.edges[c] = qs
        for c in numeric + categorical:
            b = self._bin(c, X[c])
            tab = pd.crosstab(b, y)
            tab = tab.reindex(columns=[0, 1], fill_value=0)
            good = tab[0] + self.smoothing
            bad = tab[1] + self.smoothing
            pg, pb = good / good.sum(), bad / bad.sum()
            w = np.log(pg / pb)
            self.woe[c] = w.to_dict()
            self.iv[c] = float(((pg - pb) * w).sum())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c, m in self.woe.items():
            b = self._bin(c, X[c])
            out[c] = pd.Series(b).map(m).fillna(0.0).to_numpy()
        return pd.DataFrame(out, index=X.index)


@dataclass
class Scorecard:
    """WOE logistic regression scaled to points: 600 points at 50:1 good:bad odds, 20 points to double the odds."""

    numeric: list[str]
    categorical: list[str]
    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0
    C: float = 1.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Scorecard:
        self.binner = WOEBinner().fit(X, y, self.numeric, self.categorical)
        W = self.binner.transform(X)
        self.features_ = list(W.columns)
        self.lr = LogisticRegression(C=self.C, max_iter=2000).fit(W.to_numpy(), y)
        self.factor = self.pdo / np.log(2)
        self.offset = self.base_score - self.factor * np.log(self.base_odds)
        return self

    def predict_proba_bad(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self.binner.transform(X)[self.features_].to_numpy())[:, 1]

    def feature_points(self, X: pd.DataFrame) -> pd.DataFrame:
        """Points per feature; their sum is the score (higher = lower risk). With logit(bad) = b0 + sum_j b_j * woe_j, score = offset + factor * log-odds(good)."""
        W = self.binner.transform(X)[self.features_]
        n = len(self.features_)
        b = self.lr.coef_[0]
        b0 = self.lr.intercept_[0]
        # logit(bad) = b0 + sum b_j w_j ; log-odds(good) = -logit(bad)
        pts = -(W.to_numpy() * b + b0 / n) * self.factor + self.offset / n
        return pd.DataFrame(pts, columns=self.features_, index=X.index)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.feature_points(X).sum(axis=1).to_numpy()

    def max_points(self) -> dict[str, float]:
        n = len(self.features_)
        b0 = self.lr.intercept_[0]
        out = {}
        for j, c in enumerate(self.features_):
            vals = np.array(list(self.binner.woe[c].values()))
            out[c] = float(np.max(-(vals * self.lr.coef_[0][j] + b0 / n) * self.factor + self.offset / n))
        return out

    def reason_codes(self, X: pd.DataFrame, k: int = 4) -> np.ndarray:
        """Points-below-maximum method (the classic scorecard adverse-action method): rank features by points lost."""
        P = self.feature_points(X)
        mx = self.max_points()
        lost = np.column_stack([mx[c] - P[c].to_numpy() for c in self.features_])
        order = np.argsort(-lost, axis=1, kind="stable")[:, :k]
        return np.array(self.features_, dtype=object)[order]
