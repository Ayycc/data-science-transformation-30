"""Gradient boosting (LightGBM, optional monotone constraints), decision cutoffs, performance metrics and PSI."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from .data import DatasetSpec


def frame(df: pd.DataFrame, features: list[str], categorical: list[str]) -> pd.DataFrame:
    """Model matrix: numerics as float, categoricals as pandas category (LightGBM native categorical splits)."""
    X = df[features].copy()
    for c in features:
        if c in categorical:
            X[c] = pd.Categorical(X[c].astype(str))
        else:
            X[c] = X[c].astype(float)
    return X


@dataclass
class GBMConfig:
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    monotone: bool = False
    seed: int = 26


def make_gbm(cfg: GBMConfig, features: list[str], spec: DatasetSpec) -> lgb.LGBMClassifier:
    mono = [int(spec.monotone.get(f, 0)) if f not in spec.categorical else 0 for f in features] if cfg.monotone else None
    return lgb.LGBMClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=1,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        monotone_constraints=mono,
        monotone_constraints_method="advanced" if cfg.monotone else "basic",
        random_state=cfg.seed,
        n_jobs=1,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
    )


@dataclass
class TrainedGBM:
    name: str
    model: lgb.LGBMClassifier
    features: list[str]
    categorical: list[str]
    categories: dict[str, list[str]]

    def X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = frame(df, self.features, self.categorical)
        for c in self.categorical:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.categories[c])
        return X

    def p_bad(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.X(df))[:, 1]


def train_gbm(
    name: str, train: pd.DataFrame, features: list[str], spec: DatasetSpec, cfg: GBMConfig, sample_weight: np.ndarray | None = None
) -> TrainedGBM:
    cats = [c for c in features if c in spec.categorical]
    X = frame(train, features, cats)
    categories = {c: list(X[c].cat.categories) for c in cats}
    m = make_gbm(cfg, features, spec)
    m.fit(X, train["bad"].to_numpy(), sample_weight=sample_weight)
    return TrainedGBM(name, m, features, cats, categories)


class GoodScoreModel(BaseEstimator, ClassifierMixin):
    """Prefit wrapper exposing P(good) as ``predict_proba[:, 1]`` so Fairlearn's ThresholdOptimizer can post-process it."""

    def __init__(self, scorer=None):
        self.scorer = scorer

    def fit(self, X, y=None):  # already fitted; kept for the sklearn API
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        pg = 1.0 - self.scorer(X)
        return np.column_stack([1.0 - pg, pg])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def cutoff_for_approval(p_bad: np.ndarray, approval_rate: float) -> float:
    """Approve when p_bad <= cutoff; cutoff is the approval-rate quantile of the (validation) score distribution."""
    return float(np.quantile(p_bad, approval_rate))


def ks_stat(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def performance(y: np.ndarray, p_bad: np.ndarray, approve: np.ndarray) -> dict:
    auc = roc_auc_score(y, p_bad)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_stat(y, p_bad),
        "brier": float(brier_score_loss(y, p_bad)),
        "approval_rate": float(approve.mean()),
        "bad_rate_approved": float(y[approve == 1].mean()) if approve.sum() else float("nan"),
        "bad_rate_declined": float(y[approve == 0].mean()) if (approve == 0).sum() else float("nan"),
    }


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index with quantile bins from the expected (development) distribution."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, expected, side="right"), minlength=len(edges) + 1) / len(expected)
    a = np.bincount(np.searchsorted(edges, actual, side="right"), minlength=len(edges) + 1) / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected: pd.Series, actual: pd.Series, eps: float = 1e-4) -> float:
    lv = sorted(set(expected.astype(str)) | set(actual.astype(str)))
    e = expected.astype(str).value_counts(normalize=True).reindex(lv, fill_value=0).to_numpy()
    a = actual.astype(str).value_counts(normalize=True).reindex(lv, fill_value=0).to_numpy()
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_band(v: float) -> str:
    return "stable" if v < 0.10 else ("monitor" if v < 0.25 else "action")
