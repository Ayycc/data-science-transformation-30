"""Decision-focused forecast: the inventory band each station needs for the next rebalancing horizon.

The router does not consume hourly flows directly; it needs, per station and shift, how far the
cumulative net flow will dip (``drain``: bikes that must already be there) and how far it will rise
(``fill``: docks that must be free). The extremes of a mean hourly path understate the mean of the
extremes (Jensen), so these two quantities are forecast directly with LightGBM on station-shift rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .features import STATIC_FEATURES, weights_matrix
from .forecast import LGB_PARAMS

TARGETS = ("drain", "fill")


def band_table(
    net: np.ndarray,
    hours: pd.DatetimeIndex,
    weather: pd.DataFrame,
    stations: pd.DataFrame,
    neighbours: list[np.ndarray],
    shifts=C.FLEET.shifts,
) -> pd.DataFrame:
    """One row per station x day x shift with realised drain/fill and day-ahead features."""
    S, T = net.shape
    days = pd.date_range(hours[0], hours[-1].normalize(), freq="D")
    W = weights_matrix(neighbours)
    w = weather.reindex(hours)
    frames = []
    for k, (name, h0, _, h1) in enumerate(shifts):
        drain = np.zeros((S, len(days)), dtype=np.float32)
        fill = np.zeros_like(drain)
        temp = np.zeros(len(days))
        rain = np.zeros(len(days))
        for d in range(len(days)):
            sl = slice(d * 24 + h0, d * 24 + h1)
            cum = np.cumsum(net[:, sl], axis=1)
            drain[:, d] = -np.minimum(0, cum.min(axis=1))
            fill[:, d] = np.maximum(0, cum.max(axis=1))
            temp[d] = w["temp_c"].iloc[sl].mean()
            rain[d] = w["precip_mm"].iloc[sl].sum()
        offday = ((days.dayofweek >= 5) | days.isin(list(C.HOLIDAYS))).astype(int)
        feats: dict[str, np.ndarray] = {}
        for tname, arr in (("drain", drain), ("fill", fill)):
            lag1 = np.full_like(arr, np.nan)
            lag1[:, 1:] = arr[:, :-1]
            lag7 = np.full_like(arr, np.nan)
            lag7[:, 7:] = arr[:, :-7]
            roll7 = np.full_like(arr, np.nan)
            same_type = np.full_like(arr, np.nan)  # expanding mean over past days of the same day type
            for d in range(1, len(days)):
                roll7[:, d] = arr[:, max(0, d - 7) : d].mean(axis=1)
                past = [j for j in range(d) if offday[j] == offday[d]]
                if past:
                    same_type[:, d] = arr[:, past].mean(axis=1)
            feats[f"{tname}_lag1"] = lag1
            feats[f"{tname}_lag7"] = lag7
            feats[f"{tname}_roll7"] = roll7
            feats[f"{tname}_past_daytype"] = same_type
            feats[f"nb_{tname}_past_daytype"] = (W @ np.nan_to_num(same_type)).astype(np.float32)
            feats[f"nb_{tname}_roll7"] = (W @ np.nan_to_num(roll7)).astype(np.float32)
        df = pd.DataFrame(
            {
                "station": np.repeat(np.arange(S), len(days)),
                "date": np.tile(days, S),
                "shift": name,
                "shift_id": k,
                "dow": np.tile(days.dayofweek, S),
                "offday": np.tile(offday, S),
                "h_temp": np.tile(temp, S),
                "h_rain": np.tile(rain, S),
                "drain": drain.ravel(),
                "fill": fill.ravel(),
                **{f: v.ravel() for f, v in feats.items()},
            }
        )
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    st = stations.reset_index(drop=True)
    for f in STATIC_FEATURES:
        out[f] = st[f].to_numpy(np.float32)[out["station"].to_numpy()]
    return out


BAND_FEATURES = ["shift_id", "dow", "offday", "h_temp", "h_rain", *STATIC_FEATURES] + [
    f"{p}{t}{s}"
    for t in TARGETS
    for p, s in (("", "_lag1"), ("", "_lag7"), ("", "_roll7"), ("", "_past_daytype"), ("nb_", "_past_daytype"), ("nb_", "_roll7"))
]


@dataclass
class BandModel:
    models: dict[str, lgb.Booster]
    iterations: dict[str, int]

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({t: np.clip(self.models[t].predict(table[BAND_FEATURES]), 0, None) for t in TARGETS}, index=table.index)


def fit_band(table: pd.DataFrame, fit: np.ndarray, valid: np.ndarray) -> BandModel:
    models, iters = {}, {}
    X = table[BAND_FEATURES]
    params = {**LGB_PARAMS, "objective": "regression", "metric": "l2", "min_data_in_leaf": 50, "num_leaves": 31}
    for t in TARGETS:
        m = lgb.train(
            params,
            lgb.Dataset(X[fit], table[t][fit]),
            3000,
            valid_sets=[lgb.Dataset(X[valid], table[t][valid])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        iters[t] = int(m.best_iteration)
        models[t] = lgb.train(params, lgb.Dataset(X[fit | valid], table[t][fit | valid]), max(50, int(m.best_iteration * 1.1)))
    return BandModel(models=models, iterations=iters)


def band_baseline(table: pd.DataFrame, train: np.ndarray) -> pd.DataFrame:
    """Historical-average band: station x shift x day-type mean over the training window."""
    keys = ["station", "shift_id", "offday"]
    means = table.loc[train].groupby(keys)[list(TARGETS)].mean()
    idx = pd.MultiIndex.from_frame(table[keys])
    return pd.DataFrame(means.reindex(idx).fillna(0).to_numpy(), columns=list(TARGETS), index=table.index)
