"""Day-ahead hourly station net-flow forecasts (LightGBM) against a historical-average baseline."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURES

LGB_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": C.SEED,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "verbose": -1,
}


def historical_average(panel: pd.DataFrame, train_mask: np.ndarray) -> pd.Series:
    """Baseline: each station's mean net flow for the same hour-of-week over the training window."""
    how = panel["dow"] * 24 + panel["hour"]
    tr = panel.loc[train_mask]
    table = tr.groupby([tr["station"], how[train_mask]])["net"].mean()
    key = pd.MultiIndex.from_arrays([panel["station"], how])
    return pd.Series(table.reindex(key).fillna(0.0).to_numpy(), index=panel.index, name="ha")


@dataclass
class ForecastResult:
    model: lgb.Booster
    objective: str
    best_iteration: int
    validation: dict
    importance: pd.Series

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return self.model.predict(panel[FEATURES])


def fit_lgbm(
    panel: pd.DataFrame, fit_mask: np.ndarray, valid_mask: np.ndarray, objectives: tuple[str, ...] = ("regression", "huber")
) -> ForecastResult:
    """Select objective + iteration count on the validation week, then refit on fit+valid rows."""
    X, y = panel[FEATURES], panel["net"]
    best: tuple[float, str, int] | None = None
    validation = {}
    for obj in objectives:
        params = {**LGB_PARAMS, "objective": obj, "metric": "l1"}
        dtr = lgb.Dataset(X[fit_mask], y[fit_mask], free_raw_data=False)
        dva = lgb.Dataset(X[valid_mask], y[valid_mask], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        mae = float(np.abs(m.predict(X[valid_mask], num_iteration=m.best_iteration) - y[valid_mask]).mean())
        validation[obj] = {"valid_mae": round(mae, 5), "best_iteration": int(m.best_iteration)}
        if best is None or mae < best[0]:
            best = (mae, obj, int(m.best_iteration))
    assert best is not None
    _, obj, n_iter = best
    all_mask = fit_mask | valid_mask
    final = lgb.train({**LGB_PARAMS, "objective": obj}, lgb.Dataset(X[all_mask], y[all_mask]), num_boost_round=max(50, int(n_iter * 1.1)))
    imp = pd.Series(final.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    return ForecastResult(model=final, objective=obj, best_iteration=n_iter, validation=validation, importance=imp / imp.sum())


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def evaluate(panel: pd.DataFrame, test_mask: np.ndarray, preds: dict[str, np.ndarray]) -> dict:
    """Station-hour MAE / RMSE for each forecaster, plus per-station win rate against the baseline."""
    te = panel.loc[test_mask, ["station", "net", "hour", "offday"]].copy()
    out: dict = {"n_station_hours": int(len(te)), "models": {}}
    for name, p in preds.items():
        te[name] = p[test_mask]
        err = te[name] - te["net"]
        out["models"][name] = {
            "mae": round(float(err.abs().mean()), 4),
            "rmse": round(float(np.sqrt((err**2).mean())), 4),
            "mae_weekday_peaks": round(float(err[(te["offday"] == 0) & te["hour"].isin([7, 8, 9, 16, 17, 18])].abs().mean()), 4),
        }
    per_station = te.groupby("station").apply(lambda g: pd.Series({k: mae(g[k], g["net"]) for k in preds}), include_groups=False)
    out["per_station_mae"] = per_station
    if "lgbm" in preds and "historical_average" in preds:
        win = per_station["lgbm"] < per_station["historical_average"]
        out["lgbm_beats_ha_station_share"] = round(float(win.mean()), 4)
        out["mae_improvement_vs_ha_pct"] = round(100 * (1 - out["models"]["lgbm"]["mae"] / out["models"]["historical_average"]["mae"]), 2)
    return out


def horizon_error(panel: pd.DataFrame, test_mask: np.ndarray, preds: dict[str, np.ndarray], shifts=C.FLEET.shifts) -> dict:
    """Error of the cumulative net flow over each rebalancing horizon (what the router actually consumes)."""
    te = panel.loc[test_mask, ["station", "time", "hour", "net"]].copy()
    for k, p in preds.items():
        te[k] = p[test_mask]
    te["date"] = te["time"].dt.normalize()
    out = {}
    for name, start_h, _, end_h in shifts:
        w = te[(te["hour"] >= start_h) & (te["hour"] < end_h)]
        g = w.groupby(["station", "date"])[["net", *preds]].sum()
        out[name] = {k: round(mae(g[k], g["net"]), 4) for k in preds}
    return out
