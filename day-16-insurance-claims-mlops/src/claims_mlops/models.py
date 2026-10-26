"""Candidate model pipelines (GLM Tweedie, LightGBM Tweedie) and insurance pricing metrics."""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, OrdinalEncoder, StandardScaler

from claims_mlops import config as C


def _log_density(x):
    return np.log(np.asarray(x, dtype=float))


def build_glm(alpha: float = 1e-3, features: list[str] | None = None, n_bins: int = 8) -> Pipeline:
    """Classic pricing GLM: log link Tweedie, binned numerics + one-hot categoricals."""
    feats = features or C.FEATURES
    nums = [f for f in C.NUM_FEATURES if f in feats and f != "Density"]
    cats = [f for f in C.CAT_FEATURES if f in feats]
    transformers = []
    if nums:
        transformers.append(
            ("bins", KBinsDiscretizer(n_bins=n_bins, encode="onehot", strategy="quantile", quantile_method="averaged_inverted_cdf"), nums)
        )
    if "Density" in feats:
        transformers.append(("dens", Pipeline([("log", FunctionTransformer(_log_density)), ("sc", StandardScaler())]), ["Density"]))
    if cats:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cats))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline(
        [("pre", pre), ("glm", TweedieRegressor(power=C.TWEEDIE_POWER, link="log", alpha=alpha, max_iter=1000, solver="newton-cholesky"))]
    )


def build_gbm(n_estimators: int = 350, learning_rate: float = 0.03, num_leaves: int = 15, seed: int = C.SEED) -> tuple[Pipeline, list[int]]:
    """LightGBM with the Tweedie objective; categoricals ordinal-encoded and declared categorical."""
    n_num = len(C.NUM_FEATURES)
    pre = ColumnTransformer(
        [
            ("num", "passthrough", C.NUM_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), C.CAT_FEATURES),
        ]
    )
    gbm = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=C.TWEEDIE_POWER,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=400,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    return Pipeline([("pre", pre), ("gbm", gbm)]), list(range(n_num, n_num + len(C.CAT_FEATURES)))


class BalancedModel(BaseEstimator, RegressorMixin):
    """Wraps a pipeline and rescales predictions so total predicted cost equals total actual cost on
    the training data (the actuarial balance property). LightGBM's Tweedie leaves are biased low on
    heavy-tailed claims; GLMs with a log link and intercept are already balanced (factor ~ 1)."""

    def __init__(self, pipeline: Pipeline, cat_idx: list[int] | None = None):
        self.pipeline = pipeline
        self.cat_idx = cat_idx

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> BalancedModel:
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        self.pipeline_ = clone(self.pipeline)
        last = self.pipeline_.steps[-1][0]
        kw = {f"{last}__sample_weight": w}
        if self.cat_idx is not None:
            kw[f"{last}__categorical_feature"] = self.cat_idx
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Bins whose width are too small")
            self.pipeline_.fit(X[C.FEATURES], y, **kw)
        raw = np.clip(self.pipeline_.predict(X[C.FEATURES]), 1e-6, None)
        self.balance_factor_ = float((np.asarray(y) * w).sum() / (raw * w).sum())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.pipeline_.predict(X[C.FEATURES]), 1e-6, None) * self.balance_factor_


def fit(model: Pipeline, df: pd.DataFrame, cat_idx: list[int] | None = None) -> BalancedModel:
    """Fit a balanced wrapper around ``model`` on ``df`` (exposure-weighted pure premium)."""
    return BalancedModel(model, cat_idx).fit(df[C.FEATURES], df[C.TARGET].to_numpy(), df[C.WEIGHT].to_numpy())


def gini(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    """Exposure-weighted normalised Gini from the ordered Lorenz curve (pricing convention)."""

    def _raw(score: np.ndarray) -> float:
        order = np.argsort(score, kind="mergesort")
        cw = np.cumsum(w[order]) / w.sum()
        cl = np.cumsum((y * w)[order]) / (y * w).sum()
        cw, cl = np.r_[0, cw], np.r_[0, cl]
        return float(1 - 2 * np.trapezoid(cl, cw))

    return _raw(pred) / _raw(y)


def evaluate(model, df: pd.DataFrame) -> dict[str, float]:
    """Holdout metrics. Lower deviance is better; A/E = actual / expected claim cost."""
    y, w = df[C.TARGET].to_numpy(), df[C.WEIGHT].to_numpy()
    p = np.clip(model.predict(df[C.FEATURES]), 1e-6, None)
    return {
        "tweedie_deviance": float(mean_tweedie_deviance(y, p, sample_weight=w, power=C.TWEEDIE_POWER)),
        "gini": float(gini(y, p, w)),
        "ae_ratio": float((y * w).sum() / (p * w).sum()),
        "mae": float(np.average(np.abs(y - p), weights=w)),
        "n_rows": int(len(df)),
        "exposure": float(w.sum()),
    }


def null_deviance(train: pd.DataFrame, df: pd.DataFrame) -> float:
    """Deviance of the exposure-weighted mean pure premium of ``train`` applied to ``df``."""
    mu = float(np.average(train[C.TARGET], weights=train[C.WEIGHT]))
    y, w = df[C.TARGET].to_numpy(), df[C.WEIGHT].to_numpy()
    return float(mean_tweedie_deviance(y, np.full(len(y), mu), sample_weight=w, power=C.TWEEDIE_POWER))
