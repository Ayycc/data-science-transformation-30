"""LightGBM ETA model (minutes) and its metrics."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .joins import ALL_FEATURES

MODEL_FEATURES = ["distance_km", "zone_id", "dropoff_zone_id", *ALL_FEATURES]
CATEGORICAL = ["zone_id", "dropoff_zone_id"]
PARAMS = {
    "objective": "l1",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 4,
    "deterministic": True,
    "force_col_wise": True,
}


def _x(df: pd.DataFrame) -> pd.DataFrame:
    x = df[MODEL_FEATURES].copy()
    for c in MODEL_FEATURES:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    for c in CATEGORICAL:  # small non-negative ints; LightGBM treats them as categories
        x[c] = x[c].astype("int64")
    return x


def train(df: pd.DataFrame, y: pd.Series, seed: int, num_rounds: int = 600) -> lgb.Booster:
    params = {**PARAMS, "seed": seed}
    ds = lgb.Dataset(_x(df), label=y.to_numpy(dtype=float), categorical_feature=CATEGORICAL, free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=num_rounds)


def predict(model: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    return model.predict(_x(df))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    return {
        "mae_min": float(np.mean(np.abs(err))),
        "rmse_min": float(np.sqrt(np.mean(err**2))),
        "mape_pct": float(np.mean(np.abs(err) / y) * 100.0),
        "p90_abs_err_min": float(np.percentile(np.abs(err), 90)),
        "within_2min_pct": float(np.mean(np.abs(err) <= 2.0) * 100.0),
        "n": int(len(y)),
    }
