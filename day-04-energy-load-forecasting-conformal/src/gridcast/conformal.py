"""Rolling (daily re-calibrated) Mondrian split conformal intervals with MAPIE.

The LightGBM models are trained once per backtest fold. Their out-of-sample predictions on the
calibration window and on the fold are cached, and MAPIE turns them into 90% intervals:

* ``SplitCP``: absolute-residual split conformal around the L2 point forecast
  (``mapie.regression.SplitConformalRegressor``),
* ``CQR``: conformalised quantile regression around the 5%/95% quantile models
  (``mapie.regression.ConformalizedQuantileRegressor``).

Calibration is Mondrian by region and, in ``rolling`` mode, is redone at every daily forecast origin
using the realised errors of the previous ``window_days`` days (all of which are out-of-sample for the
model, because it was trained before the calibration window began). ``static`` mode calibrates once
at the fold origin, the textbook split-conformal set-up.

Scores are computed in the level-normalised target space (load / trailing 7-day mean) and the bounds
are rescaled to MW, so the interval width tracks the regional load level.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from mapie.regression import ConformalizedQuantileRegressor, SplitConformalRegressor
from sklearn.base import BaseEstimator, RegressorMixin

from .config import LEVEL

# MAPIE switches the root logger to INFO and reports crossing CQR quantiles as "ill-sorted"; it then
# sorts them itself. Keep that out of the run log.
logging.getLogger().addFilter(lambda record: "ill-sorted" not in record.getMessage())

PRED_COLS = ["p_point", "p_lo", "p_hi", "p_med"]


class CachedPrediction(RegressorMixin, BaseEstimator):
    """A prefit regressor whose prediction is a pre-computed column of X.

    Lets MAPIE conformalise forecasts that were already produced by the trained LightGBM models,
    without re-running the boosted trees for every daily recalibration.
    """

    def __init__(self, column: int = 0):
        self.column = column

    def fit(self, X, y=None):  # noqa: D401 - sklearn API
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.asarray(X)[:, self.column]


def _estimators():
    return [CachedPrediction(i).fit(None) for i in range(len(PRED_COLS))]


def conformalise(
    raw: pd.DataFrame,
    fold_start: pd.Timestamp,
    method: str,
    window_days: int,
    mode: str = "rolling",
    aci_gamma: float | None = None,
    block_hours: int = 24,
) -> pd.DataFrame:
    """Turn cached model predictions into interval forecasts for the fold.

    raw: columns region, ds, horizon, y_rel, level, load_mw and PRED_COLS, covering the calibration
         window before the fold and the fold itself.
    mode: "rolling" recalibrates at every daily origin on the previous `window_days`; "static"
          calibrates once at the fold origin on the `window_days` before it.
    aci_gamma: if set (rolling mode only), Adaptive Conformal Inference (Gibbs and Candes, 2021):
          the miscoverage level used for day t+1 is alpha_{t+1} = alpha_t + gamma * (alpha - miss_t),
          where miss_t is the realised miss rate of day t. Updates start as soon as a full window of
          calibration history exists (inside the pre-fold calibration window), so the fold starts warm.
    block_hours: Mondrian groups are region x block of `block_hours` consecutive horizons
          (24 = one group per region; 6 = four groups per region: night, morning, afternoon, evening).
          Each group gets its own calibration set, quantile and ACI state.
    Returns: region, ds, horizon, y, q05, q50, q95 (MW) for rows with ds >= fold_start, plus alpha_used.
    """
    if method not in ("SplitCP", "CQR"):
        raise ValueError(method)
    if mode not in ("rolling", "static"):
        raise ValueError(mode)
    if aci_gamma is not None and mode != "rolling":
        raise ValueError("ACI needs rolling mode")
    if 24 % block_hours:
        raise ValueError("block_hours must divide 24")
    point, lo, hi, med = _estimators()
    target_alpha = 1 - LEVEL / 100
    out = []
    raw = raw.assign(day=raw["ds"].dt.normalize(), block=(raw["horizon"] - 1) // block_hours)
    first_day = raw["day"].min()
    for (region, _), g in raw.groupby(["region", "block"], sort=False):
        days = sorted(g["day"].unique())
        if mode == "static":
            origins = [fold_start]
        elif aci_gamma is None:
            origins = [d for d in days if d >= fold_start]
        else:
            origins = [d for d in days if d >= first_day + pd.Timedelta(days=window_days)]
        alpha_t = target_alpha
        for day in origins:
            cal = g[(g["day"] >= day - pd.Timedelta(days=window_days)) & (g["day"] < day)]
            tgt = g[g["day"] >= day] if mode == "static" else g[g["day"] == day]
            Xc, yc = cal[PRED_COLS].to_numpy(float), cal["y_rel"].to_numpy(float)
            # MAPIE needs 1 / alpha < n_calibration (2 / alpha for asymmetric CQR); cap the ACI level.
            conf = float(np.clip(1 - alpha_t, 0.5, min(0.995, 1 - 2.5 / len(yc))))
            Xt = tgt[PRED_COLS].to_numpy(float)
            if method == "SplitCP":
                m = SplitConformalRegressor(point, confidence_level=conf, conformity_score="absolute", prefit=True)
            else:
                m = ConformalizedQuantileRegressor([lo, hi, med], confidence_level=conf, prefit=True)
            m.conformalize(Xc, yc)
            yp, pis = m.predict_interval(Xt)
            lvl = tgt["level"].to_numpy(float)
            y = tgt["load_mw"].to_numpy()
            q05, q95 = pis[:, 0, 0] * lvl, pis[:, 1, 0] * lvl
            if aci_gamma is not None:
                miss = float(np.mean((y < q05) | (y > q95)))
                alpha_t = alpha_t + aci_gamma * (target_alpha - miss)
            if day < fold_start:
                continue
            out.append(
                pd.DataFrame(
                    {
                        "region": region,
                        "ds": tgt["ds"].to_numpy(),
                        "horizon": tgt["horizon"].to_numpy(),
                        "y": y,
                        "q05": q05,
                        "q50": yp * lvl,
                        "q95": q95,
                        "alpha_used": 1 - conf,
                    }
                )
            )
    return pd.concat(out, ignore_index=True).sort_values(["region", "ds"], kind="stable").reset_index(drop=True)
