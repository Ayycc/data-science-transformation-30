"""Point and quantile demand forecasts with statsforecast (ETS, Croston-SBA, TSB) plus a holdout evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsforecast import StatsForecast
from statsforecast.models import TSB, AutoETS, CrostonSBA, Naive, WindowAverage

MODEL_NAMES = ("Naive", "Planner13wk", "AutoETS", "CrostonSBA", "TSB")


def _models() -> list:
    return [
        Naive(),
        WindowAverage(window_size=13, alias="Planner13wk"),
        AutoETS(season_length=1, model="ZZN", alias="AutoETS"),
        CrostonSBA(),
        TSB(alpha_d=0.1, alpha_p=0.1),
    ]


def to_sf(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"item_id": "unique_id", "week": "ds", "demand": "y"})[["unique_id", "ds", "y"]].astype({"y": float})


def forecast_all(history: pd.DataFrame, h: int) -> pd.DataFrame:
    """Forecast every item with every model; returns long frame unique_id, ds, <model columns>."""
    sf = StatsForecast(models=_models(), freq=1, n_jobs=1, fallback_model=Naive())
    fc = sf.forecast(df=to_sf(history), h=h)
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    for m in MODEL_NAMES:
        fc[m] = fc[m].clip(lower=0.0)
    return fc


def holdout_evaluation(
    history: pd.DataFrame, seg: pd.DataFrame, val_weeks: int, non_seasonal: pd.Index
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on history[:-val_weeks], score each model on the last ``val_weeks``.

    Returns per-item errors (scaled MAE = MAE / mean training demand, and RMSE) and a per-SBC-class summary.
    """
    n = int(history["week"].max()) + 1
    cut = n - val_weeks
    hist = history[history["item_id"].isin(non_seasonal)]
    train, val = hist[hist["week"] < cut], hist[hist["week"] >= cut]
    fc = forecast_all(train, val_weeks)
    merged = fc.merge(to_sf(val), on=["unique_id", "ds"], how="inner")
    scale = train.groupby("item_id")["demand"].mean().clip(lower=1e-6)
    rows = []
    for m in MODEL_NAMES:
        err = merged[m] - merged["y"]
        g = err.groupby(merged["unique_id"])
        per = pd.DataFrame({"mae": g.apply(lambda e: e.abs().mean()), "rmse": g.apply(lambda e: np.sqrt((e**2).mean()))})
        per["bias"] = g.mean()
        per["smae"] = per["mae"] / scale.reindex(per.index)
        per["srmse"] = per["rmse"] / scale.reindex(per.index)
        per["model"] = m
        rows.append(per)
    per_item = pd.concat(rows).rename_axis("item_id").reset_index()
    per_item = per_item.merge(seg[["sbc", "abc"]], left_on="item_id", right_index=True)
    summary = (
        per_item.groupby(["sbc", "model"])
        .agg(smae=("smae", "mean"), srmse=("srmse", "mean"), bias=("bias", "mean"), n_items=("item_id", "nunique"))
        .reset_index()
    )
    return per_item, summary


def select_models(summary: pd.DataFrame) -> dict[str, str]:
    """Pick the model with the lowest mean scaled RMSE in each SBC class (Naive and the planner rule excluded).

    RMSE, not MAE, because the policies need an unbiased *mean*: MAE rewards the median, which is ~0 for
    intermittent demand.
    """
    cand = summary[summary["model"].isin(["AutoETS", "CrostonSBA", "TSB"])]
    best = cand.loc[cand.groupby("sbc")["srmse"].idxmin()]
    return dict(zip(best["sbc"], best["model"]))


def final_forecasts(
    history: pd.DataFrame, seg: pd.DataFrame, choice: dict[str, str], per_item: pd.DataFrame, h: int = 8
) -> pd.DataFrame:
    """Refit on the full history; mean weekly forecast (avg of h steps) and holdout RMSE of the chosen model."""
    items = seg.index[~seg["seasonal"]]
    fc = forecast_all(history[history["item_id"].isin(items)], h)
    out = pd.DataFrame(index=items)
    out["sbc"] = seg.loc[items, "sbc"]
    out["model"] = out["sbc"].map(choice).fillna("AutoETS")
    means = fc.groupby("unique_id")[list(MODEL_NAMES)].mean()
    out["mu"] = [means.at[i, m] for i, m in zip(out.index, out["model"])]
    rm = per_item.set_index(["item_id", "model"])["rmse"]
    out["sigma"] = [rm.get((i, m), np.nan) for i, m in zip(out.index, out["model"])]
    out["sigma"] = out["sigma"].fillna(out["mu"].clip(lower=0.1) ** 0.5)
    out["dist"] = np.where(out["sbc"] == "smooth", "normal", "gamma")
    return out


def lead_time_demand(
    mu: np.ndarray, sigma: np.ndarray, lt_mean: np.ndarray, lt_sd: np.ndarray, review: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd of demand over the protection period L + R with random L (standard compound formula)."""
    p = lt_mean + review
    mean = mu * p
    var = p * sigma**2 + mu**2 * lt_sd**2
    return mean, np.sqrt(var)


def quantile(mean: np.ndarray, sd: np.ndarray, q: np.ndarray | float, dist: np.ndarray) -> np.ndarray:
    """Quantile of a normal (smooth demand) or gamma (intermittent / erratic, non-negative and skewed) law."""
    mean = np.asarray(mean, float)
    sd = np.maximum(np.asarray(sd, float), 1e-9)
    q = np.broadcast_to(np.asarray(q, float), mean.shape)
    normal = np.maximum(stats.norm.ppf(q, loc=mean, scale=sd), 0.0)
    shape = np.where(mean > 0, (mean / sd) ** 2, 1.0)
    scale = np.where(mean > 0, sd**2 / np.maximum(mean, 1e-9), 1.0)
    gam = np.where(mean > 0, stats.gamma.ppf(q, a=shape, scale=scale), 0.0)
    return np.where(np.asarray(dist) == "normal", normal, gam)


def pinball(y: np.ndarray, qhat: np.ndarray, q: float) -> float:
    diff = y - qhat
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))
