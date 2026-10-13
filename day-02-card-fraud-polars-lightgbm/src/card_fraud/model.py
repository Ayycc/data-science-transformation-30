"""Time-based split, LightGBM + Optuna, logistic-regression challenger, isotonic calibration."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from card_fraud.config import CATEGORICAL_FEATURES, FEATURES, SEED, TEST_FOLDS, TRAIN_FOLDS, VAL_FOLDS

HEAVY_TAILED = (
    "amount", "card_amount_sum_1h", "card_amount_sum_24h", "card_amount_mean_30d", "amount_to_mean_30d",
    "secs_since_prev", "dist_prev_km", "speed_kmh", "dist_from_centroid_km", "card_txn_count_7d",
    "device_prior_txn_count", "merchant_hist_txn_count", "card_txn_index",
)  # fmt: skip


def time_split(feats: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Calendar-month split: train Jan-Apr, validation May, test June."""
    return {
        "train": feats.filter(pl.col("fold").is_in(TRAIN_FOLDS)),
        "val": feats.filter(pl.col("fold").is_in(VAL_FOLDS)),
        "test": feats.filter(pl.col("fold").is_in(TEST_FOLDS)),
    }


def downsample_negatives(df: pl.DataFrame, rate: float, seed: int = SEED) -> pl.DataFrame:
    """Keep every fraud row and a seeded random share of legit rows."""
    if rate >= 1.0:
        return df
    rng = np.random.default_rng(seed)
    keep = (df["is_fraud"].to_numpy() == 1) | (rng.random(df.height) < rate)
    return df.filter(pl.Series(keep))


def to_xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df.select([pl.col(c).cast(pl.Float64) for c in FEATURES]).to_numpy()
    y = df["is_fraud"].to_numpy().astype(np.int32)
    return x, y


BASE_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "verbosity": -1,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "bagging_freq": 1,
}


def _dataset(x: np.ndarray, y: np.ndarray, ref: lgb.Dataset | None = None) -> lgb.Dataset:
    cat = [FEATURES.index(c) for c in CATEGORICAL_FEATURES]
    return lgb.Dataset(x, y, feature_name=list(FEATURES), categorical_feature=cat, reference=ref, free_raw_data=False)


def fit_lgbm(
    x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray, params: dict, max_rounds: int = 800
) -> lgb.Booster:
    dtr = _dataset(x_tr, y_tr)
    dva = _dataset(x_va, y_va, ref=dtr)
    return lgb.train(
        {**BASE_PARAMS, **params},
        dtr,
        num_boost_round=max_rounds,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )


def tune_lgbm(
    x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray, n_trials: int, seed: int = SEED
) -> tuple[dict, optuna.Study]:
    """Optuna TPE search maximising validation PR-AUC (early stopping on validation)."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.25, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 400, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0, log=True),
        }
        booster = fit_lgbm(x_tr, y_tr, x_va, y_va, params)
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        return float(average_precision_score(y_va, booster.predict(x_va, num_iteration=booster.best_iteration)))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params), study


def fit_logreg(x_tr: np.ndarray, y_tr: np.ndarray, seed: int = SEED) -> Pipeline:
    """Challenger: log-transformed, standardised features + one-hot MCC, class-balanced logistic regression."""
    cat_idx = [FEATURES.index(c) for c in CATEGORICAL_FEATURES]
    log_idx = [FEATURES.index(c) for c in HEAVY_TAILED]
    lin_idx = [i for i in range(len(FEATURES)) if i not in cat_idx and i not in log_idx]
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    logn = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(lambda a: np.log1p(np.clip(a, 0, None)))),
            ("scale", StandardScaler()),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", num, lin_idx),
            ("log", logn, log_idx),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_idx),
        ]
    )
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)]).fit(x_tr, y_tr)


def fit_isotonic(score_val: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(score_val, y_val)
