"""Readmission models: regularised logistic regression, calibrated gradient boosting, LACE baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

SPLINE = ["age", "hgb_last", "hgb_min", "sodium_last", "creat_last", "creat_max", "glucose_last", "wbc_last"]
COUNTS = ["los_days", "prior_ip_365d", "prior_ed_180d", "prior_amb_365d", "charlson", "n_drugs_365d"]
FLAGS = [
    "emergent",
    "elective",
    "chf",
    "copd",
    "diabetes",
    "ckd",
    "cancer",
    "dementia",
    "liver",
    "prior_mi",
    "hypertension",
    "afib",
    "depression",
]
CATEGORICAL = ["sex", "primary_dx", "disposition"]
FEATURES = SPLINE + COUNTS + FLAGS + CATEGORICAL
TARGET = "readmit_30d"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in FLAGS:
        X[c] = X[c].astype(float)
    for c in COUNTS:
        X[c] = X[c].astype(float)
    X["los_days"] = np.log1p(X["los_days"])
    for c in ("prior_ip_365d", "prior_ed_180d"):
        X[c] = X[c].clip(upper=6)
    for c in CATEGORICAL:
        X[c] = X[c].astype(str).astype("category")
    return X


def make_logistic(seed: int = 42) -> Pipeline:
    """L2-penalised logistic regression with restricted-cubic-style splines on age and labs."""
    pre = ColumnTransformer(
        [
            (
                "spline",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    SplineTransformer(n_knots=4, degree=3, extrapolation="linear"),
                    StandardScaler(),
                ),
                SPLINE,
            ),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), COUNTS + FLAGS),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
        ]
    )
    clf = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 13),
        l1_ratios=(0.0,),
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        scoring="neg_log_loss",
        max_iter=5000,
        use_legacy_attributes=False,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def make_gbm(seed: int = 42) -> CalibratedClassifierCV:
    """Histogram gradient boosting with early stopping, Platt-recalibrated by 5-fold cross-fitting."""
    gbm = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=600,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=seed,
    )
    return CalibratedClassifierCV(gbm, method="sigmoid", cv=StratifiedKFold(5, shuffle=True, random_state=seed))


def make_lace() -> Pipeline:
    """LACE score mapped to a probability by a one-variable logistic recalibration fitted on train."""
    return Pipeline([("clf", LogisticRegression(C=1e6, max_iter=1000))])


MODELS = {"logistic_l2": make_logistic, "gbm_calibrated": make_gbm}


def oof_predictions(
    name: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int = 42, n_splits: int = 5
) -> np.ndarray:
    """Out-of-fold predictions on the training period with patients kept within one fold."""
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = MODELS[name](seed)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    return pd.DataFrame({"feature": pre.get_feature_names_out(), "coef": clf.coef_.ravel()}).assign(
        C=float(np.ravel(clf.C_)[0])
    )
