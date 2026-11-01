"""SYNTHETIC field yields, yield models with random vs spatially blocked CV, and the climate-risk scenario."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from .config import LAT_MIN, LON_MIN
from .generate import basis_points

CLIMATE_FEATURES = ["gdd", "heat_days", "precip_jja", "spi3_aug", "lat", "lon", "year"]
NOWCAST_FEATURES = CLIMATE_FEATURES + ["ndvi_int", "ndvi_peak"]

# Ground-truth yield response (t/ha). Only the generator knows this; the model never sees soil or the residual.
Y_MAX = 12.0
GDD_OPT = 1750.0
GDD_WIDTH = 520.0
HEAT_COEF = 0.012
TREND_PER_YEAR = 0.08
RESID_SD = 0.55
NOISE_SD = 0.35


def yield_response(gdd, heat, spi_v, soil, year, resid):
    """Deterministic part of the SYNTHETIC yield model (vectorised)."""
    f_gdd = np.exp(-0.5 * ((np.asarray(gdd) - GDD_OPT) / GDD_WIDTH) ** 2)
    f_heat = np.exp(-HEAT_COEF * np.asarray(heat))
    f_water = 0.3 + 0.7 / (1.0 + np.exp(-1.7 * (np.asarray(spi_v) + 1.1)))
    return Y_MAX * soil * f_gdd * f_heat * f_water + TREND_PER_YEAR * (np.asarray(year) - 2001) + resid


def spatial_year_residual(lat: np.ndarray, lon: np.ndarray, years: np.ndarray, seed: int) -> np.ndarray:
    """Spatially smooth, year-specific yield residual (pests, hail, local management) at field centroids."""
    out = np.empty(len(lat))
    for y in np.unique(years):
        m = years == y
        phi = basis_points(lat[m], lon[m], seed + 3, 12, 0.8, 3.0)
        c = np.random.default_rng([seed, int(y), 3]).normal(size=12)
        out[m] = RESID_SD * (c @ phi)
    return out


def add_true_yield(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Attach the SYNTHETIC observed yield (truth + noise) to a field-year feature table."""
    df = df.copy()
    df["resid_true"] = spatial_year_residual(df["lat"].to_numpy(), df["lon"].to_numpy(), df["year"].to_numpy(), seed)
    mean = yield_response(df["gdd"], df["heat_days"], df["spi3_aug"], df["soil"], df["year"], df["resid_true"])
    noise = np.random.default_rng([seed, 11]).normal(0, NOISE_SD, len(df))
    df["yield_t_ha"] = mean + noise
    return df


def spatial_blocks(lat: np.ndarray, lon: np.ndarray, block_deg: float) -> np.ndarray:
    bi = np.floor((np.asarray(lat) - LAT_MIN) / block_deg).astype(int)
    bj = np.floor((np.asarray(lon) - LON_MIN) / block_deg).astype(int)
    return bi * 1000 + bj


def make_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)


def _scores(y, p) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
    }


def cross_validate(df: pd.DataFrame, features: list[str], scheme: str, n_folds: int, block_deg: float, seed: int) -> dict:
    """Out-of-fold scores. scheme='random' (row-level KFold) or 'spatial' (GroupKFold over lat/lon blocks)."""
    X = df[features].to_numpy()
    y = df["yield_t_ha"].to_numpy()
    if scheme == "random":
        splits = KFold(n_folds, shuffle=True, random_state=seed).split(X)
    elif scheme == "spatial":
        groups = spatial_blocks(df["lat"], df["lon"], block_deg)
        splits = GroupKFold(n_folds, shuffle=True, random_state=seed).split(X, y, groups)
    else:
        raise ValueError(scheme)
    oof = np.empty_like(y)
    for tr, te in splits:
        oof[te] = make_model(seed).fit(X[tr], y[tr]).predict(X[te])
    return _scores(y, oof) | {"scheme": scheme, "n_folds": n_folds, "features": features}


def scenario_table(base: pd.DataFrame, scen: pd.DataFrame, model, features: list[str], ref_year: int) -> pd.DataFrame:
    """Field x weather-year yields under baseline and scenario climate, at `ref_year` technology (trend)."""
    keys = ["field_id", "year"]
    b = base.set_index(keys).sort_index()
    s = scen.set_index(keys).sort_index()
    out = pd.DataFrame(index=b.index)
    for name, frame in (("base", b), ("scen", s)):
        X = frame.reset_index().assign(year=ref_year)[features].to_numpy()
        out[f"pred_{name}"] = model.predict(X)
        out[f"true_{name}"] = yield_response(frame["gdd"], frame["heat_days"], frame["spi3_aug"], b["soil"], ref_year, b["resid_true"]).to_numpy()
    return out.reset_index().rename(columns={"year": "weather_year"})


def risk_metrics(tab: pd.DataFrame, col_base: str, col_scen: str, trigger: float = 0.8) -> dict:
    """Portfolio yield distribution statistics and yield-at-risk (YaR95 = mean - 5th percentile)."""
    out = {}
    field_mean_base = tab.groupby("field_id")[col_base].transform("mean")
    for label, col in (("baseline", col_base), ("scenario", col_scen)):
        v = tab[col].to_numpy()
        p5 = float(np.percentile(v, 5))
        tail = v[v <= p5]
        out[label] = {
            "mean_t_ha": float(v.mean()),
            "p5_t_ha": p5,
            "yar95_t_ha": float(v.mean() - p5),
            "cvar95_t_ha": float(tail.mean()),
            "p_below_trigger": float((tab[col] < trigger * field_mean_base).mean()),
        }
    out["mean_change_pct"] = 100 * (out["scenario"]["mean_t_ha"] / out["baseline"]["mean_t_ha"] - 1)
    out["p5_change_pct"] = 100 * (out["scenario"]["p5_t_ha"] / out["baseline"]["p5_t_ha"] - 1)
    out["trigger_fraction_of_baseline_mean"] = trigger
    return out
