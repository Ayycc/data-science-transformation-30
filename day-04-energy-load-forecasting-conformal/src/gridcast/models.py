"""Forecasters: statsforecast baselines and a global LightGBM model with MAPIE conformal intervals."""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA, AutoETS, SeasonalNaive

from .config import HORIZON, LEVEL
from .features import CATEGORICAL, FEATURES

# Each statsforecast group: (name, models, training window in hours or None for full history, refit).
# Windows keep the classical fits laptop-sized. ETS, ARIMA and MSTL parameters are fitted once at the fold
# origin and then rolled forward day by day with `forward` (refit=False), as an operator would re-run a
# frozen model. Seasonal naive has no parameters, so it is simply re-applied at every origin (refit=True;
# statsforecast 2.0.x, used on Python 3.11, has no `forward` for it).
STATS_GROUPS: list[tuple[str, list, int | None, bool]] = [
    (
        "naive",
        [
            SeasonalNaive(season_length=24, alias="SeasonalNaive-24h"),
            SeasonalNaive(season_length=168, alias="SeasonalNaive-168h"),
        ],
        None,
        True,
    ),
    ("ets", [AutoETS(season_length=24, model="ZZA", damped=True, alias="AutoETS")], 24 * 28, False),
    (
        "arima",
        [AutoARIMA(season_length=24, max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1, max_D=1, alias="AutoARIMA")],
        24 * 14,
        False,
    ),
    ("mstl", [MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"), alias="MSTL")], 24 * 56, False),
]

LGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=4,
    verbose=-1,
)


def stats_backtest_fold(series: pd.DataFrame, fold_start: pd.Timestamp, fold_days: int, groups=None) -> pd.DataFrame:
    """Day-ahead forecasts for every day of one fold from each statsforecast model.

    series: columns unique_id, ds, y (history up to at least the fold end).
    Returns long frame: model, region, ds, horizon, y, q05, q50, q95.
    """
    fold_end = fold_start + pd.Timedelta(days=fold_days)
    d = series[series["ds"] < fold_end]
    out = []
    for _, models, input_size, refit in groups or STATS_GROUPS:
        sf = StatsForecast(models=models, freq="h", n_jobs=4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = sf.cross_validation(
                h=HORIZON,
                df=d,
                n_windows=fold_days,
                step_size=HORIZON,
                input_size=input_size,
                level=[LEVEL],
                refit=refit,
            )
        cv = cv[cv["ds"] >= fold_start]
        for m in models:
            a = m.alias
            out.append(
                pd.DataFrame(
                    {
                        "model": a,
                        "region": cv["unique_id"].to_numpy(),
                        "ds": cv["ds"].to_numpy(),
                        "horizon": ((cv["ds"] - cv["cutoff"]) / pd.Timedelta(hours=1)).astype(int).to_numpy(),
                        "y": cv["y"].to_numpy(),
                        "q05": cv[f"{a}-lo-{LEVEL}"].to_numpy(),
                        "q50": cv[a].to_numpy(),
                        "q95": cv[f"{a}-hi-{LEVEL}"].to_numpy(),
                    }
                )
            )
    return pd.concat(out, ignore_index=True)


def _fit_lgbm(X: np.ndarray, y: np.ndarray, seed: int, **extra) -> lgb.LGBMRegressor:
    cat_idx = [FEATURES.index(c) for c in CATEGORICAL]
    m = lgb.LGBMRegressor(random_state=seed, **LGBM_PARAMS, **extra)
    m.fit(X, y, categorical_feature=cat_idx)
    return m


def cached_predictions(models: dict[str, lgb.LGBMRegressor], rows: pd.DataFrame) -> pd.DataFrame:
    """Predictions of the four LightGBM models on feature rows, in the level-normalised space."""
    X = rows[FEATURES].to_numpy(float)
    raw = rows[["region", "ds", "horizon", "y_rel", "level", "load_mw"]].reset_index(drop=True)
    for col, m in models.items():
        raw[col] = m.predict(X)
    return raw


def lgbm_fold_predictions(
    feats: pd.DataFrame, fold_start: pd.Timestamp, fold_days: int, cal_days: int, seed: int
) -> tuple[pd.DataFrame, dict]:
    """Train the global LightGBM models on history before the calibration window; predict calibration + fold.

    Four models share one feature set and the level-normalised target: an L2 point model and
    5% / 95% / 50% quantile models. Training data ends `cal_days` before the fold starts, so every
    cached prediction (calibration window and fold) is out-of-sample. Returns the cached predictions
    (columns region, ds, horizon, y_rel, level, load_mw, p_point, p_lo, p_hi, p_med) and fit info
    (including the fitted models under "models").
    """
    fold_end = fold_start + pd.Timedelta(days=fold_days)
    cal_start = fold_start - pd.Timedelta(days=cal_days)
    f = feats.dropna(subset=[*FEATURES, "y_rel"])
    tr = f[f["ds"] < cal_start]
    ev = f[(f["ds"] >= cal_start) & (f["ds"] < fold_end)]
    Xtr, ytr = tr[FEATURES].to_numpy(float), tr["y_rel"].to_numpy(float)
    models = {"p_point": _fit_lgbm(Xtr, ytr, seed)}
    for col, alpha in (("p_lo", 0.05), ("p_hi", 0.95), ("p_med", 0.5)):
        models[col] = _fit_lgbm(Xtr, ytr, seed, objective="quantile", alpha=alpha)
    info = {
        "train_start": str(tr["ds"].min()),
        "train_end": str(tr["ds"].max()),
        "n_train": int(len(tr)),
        "n_cal_and_test": int(len(ev)),
        "models": models,
    }
    return cached_predictions(models, ev), info
