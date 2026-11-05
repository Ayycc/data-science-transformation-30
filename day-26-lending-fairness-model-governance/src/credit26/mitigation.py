"""Mitigation: reweighing (pre-processing), exponentiated gradient (in-processing), threshold optimisation (post-processing)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, ErrorRate, ExponentiatedGradient

from .models import GoodScoreModel


def reweighing_weights(groups: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Kamiran & Calders (2012): w(g, y) = P(g) P(y) / P(g, y), raised to ``alpha`` (0 = no reweighing, 1 = full)."""
    d = pd.DataFrame({"g": groups, "y": y})
    pg = d["g"].map(d["g"].value_counts(normalize=True))
    py = d["y"].map(d["y"].value_counts(normalize=True))
    pgy = d.groupby(["g", "y"])["y"].transform("size") / len(d)
    w = (pg * py / pgy).to_numpy()
    return w**alpha


def numeric_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """All-numeric matrix (categoricals as integer codes) for Fairlearn reductions, which require array-like input."""
    return df[features].apply(lambda s: pd.to_numeric(s, errors="coerce") if s.dtype != object else s.astype("category").cat.codes).astype(float)


def exponentiated_gradient(
    train: pd.DataFrame, features: list[str], sensitive: str, eps: float, seed: int = 26, max_iter: int = 30, fp_cost: float = 1.0
):
    """Agarwal et al. (2018) reductions approach with a demographic-parity bound on approval (favourable = good label).

    ``fp_cost`` is the cost of approving a bad relative to declining a good. Setting it to (1 - c) / c, where c is the
    unconstrained model's p_bad cutoff, makes the cost-sensitive classifiers approve at roughly the business approval rate
    instead of approving nearly everyone (the plain-error-rate optimum when most applicants are good).
    """
    base = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.08,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )
    eg = ExponentiatedGradient(
        base, constraints=DemographicParity(difference_bound=eps), objective=ErrorRate(costs={"fp": fp_cost, "fn": 1.0}), max_iter=max_iter, nu=1e-6
    )
    eg.fit(numeric_matrix(train, features), 1 - train["bad"].to_numpy(), sensitive_features=train[sensitive].astype(str).to_numpy())
    return eg


def eg_p_bad(eg, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Deterministic score = 1 - expected P(approve) under EG's randomised mixture of classifiers."""
    return 1.0 - eg._pmf_predict(numeric_matrix(df, features))[:, 1]


def threshold_optimizer(scorer, valid: pd.DataFrame, sensitive: str, constraint: str = "demographic_parity", seed: int = 26) -> ThresholdOptimizer:
    """Hardt et al. (2016) group-specific thresholds, fitted on validation. Uses the protected attribute at decision time."""
    est = GoodScoreModel(scorer).fit(None)
    to = ThresholdOptimizer(estimator=est, constraints=constraint, objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba")
    to.fit(valid, 1 - valid["bad"].to_numpy(), sensitive_features=valid[sensitive].astype(str).to_numpy())
    return to


def to_approve(to: ThresholdOptimizer, df: pd.DataFrame, sensitive: str, seed: int = 26) -> np.ndarray:
    return np.asarray(to.predict(df, sensitive_features=df[sensitive].astype(str).to_numpy(), random_state=seed)).astype(int)
