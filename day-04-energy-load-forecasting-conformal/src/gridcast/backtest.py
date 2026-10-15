"""Rolling-origin backtest: baselines, global LightGBM, conformal intervals, metrics, stress test."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFORMAL_GRID, HORIZON, LEVEL, STRESS_SCENARIOS, Scale
from .conformal import conformalise
from .features import build_features
from .generate import generate
from .metrics import coverage_table, mase_scale, summarise
from .models import STATS_GROUPS, cached_predictions, lgbm_fold_predictions, stats_backtest_fold

SELECTED = "LightGBM+conformal (selected)"


def fold_starts(scale: Scale) -> list[pd.Timestamp]:
    """Test folds: n_folds consecutive blocks of fold_days ending at the end of the data."""
    end = pd.Timestamp(scale.end)
    first = end - pd.Timedelta(days=scale.n_folds * scale.fold_days)
    return [first + pd.Timedelta(days=k * scale.fold_days) for k in range(scale.n_folds)]


def val_fold_starts(scale: Scale) -> list[pd.Timestamp]:
    s = pd.Timestamp(scale.val_start)
    return [s + pd.Timedelta(days=k * scale.val_spacing_days) for k in range(scale.n_val_folds)]


def config_name(method: str, window: int, gamma: float | None, block: int = 24) -> str:
    groups = "region" if block == 24 else f"region x {block}h block"
    return f"{method}, rolling {window}d" + (f", ACI gamma={gamma}" if gamma is not None else "") + f", by {groups}"


def apply_cfg(raw: pd.DataFrame, fs: pd.Timestamp, cfg: tuple) -> pd.DataFrame:
    method, window, gamma, block = cfg
    return conformalise(raw, fs, method, window, "rolling", gamma, block_hours=block)


def mase_scales(df: pd.DataFrame, starts: list[pd.Timestamp]) -> pd.DataFrame:
    """MASE denominator per region and fold: in-sample weekly seasonal-naive MAE over the prior 365 days."""
    rows = []
    for k, fs in enumerate(starts):
        for region, g in df.groupby("region"):
            y = g[(g["ds"] < fs) & (g["ds"] >= fs - pd.Timedelta(days=365))].sort_values("ds")["load_mw"]
            rows.append({"region": region, "fold": k, "scale": mase_scale(y.to_numpy(), m=168)})
    return pd.DataFrame(rows)


def select_conformal(feats: pd.DataFrame, scale: Scale, log) -> tuple[tuple, pd.DataFrame]:
    """Choose the conformal configuration on validation folds that precede the test year.

    Criterion: mean absolute deviation from 90% of coverage in each region x horizon cell (pooled over
    validation folds); ties broken by mean interval width. The test folds are not used.
    """
    grid = [c for c in CONFORMAL_GRID if c[1] <= scale.cal_days]  # window must fit inside the calibration span
    preds = []
    for k, fs in enumerate(val_fold_starts(scale)):
        t = time.time()
        raw, _ = lgbm_fold_predictions(feats, fs, scale.fold_days, scale.cal_days, scale.seed)
        for cfg in grid:
            p = apply_cfg(raw, fs, cfg)
            preds.append(p.assign(config=config_name(*cfg), fold=k))
        log(f"  validation fold {k} ({fs.date()}) {time.time() - t:.1f}s")
    p = pd.concat(preds, ignore_index=True)
    p["hit"] = (p["y"] >= p["q05"]) & (p["y"] <= p["q95"])
    p["width_pct"] = (p["q95"] - p["q05"]) / p["y"] * 100
    by_region = p.groupby(["config", "region"])["hit"].mean().unstack()
    by_cell = p.groupby(["config", "region", "horizon"])["hit"].mean()
    cell_err = (by_cell - LEVEL / 100).abs().groupby(level="config").mean()
    tab = pd.DataFrame(
        {
            "coverage": p.groupby("config")["hit"].mean(),
            "cell_cov_abs_err": cell_err,
            "region_cov_abs_err": (by_region - LEVEL / 100).abs().mean(axis=1),
            "min_region_cov": by_region.min(axis=1),
            "max_region_cov": by_region.max(axis=1),
            "width_pct": p.groupby("config")["width_pct"].mean(),
        }
    ).reset_index()
    tab = tab.sort_values(["cell_cov_abs_err", "width_pct"]).reset_index(drop=True)
    best = next(c for c in grid if config_name(*c) == tab.loc[0, "config"])
    return best, tab


def run_backtest(df: pd.DataFrame, scale: Scale, log=print, stress: bool = True) -> dict:
    """Run the full backtest on a load frame. Returns a dict of tables and metrics (nothing written)."""
    timings: dict[str, float] = {}
    t0 = time.time()
    feats = build_features(df)
    series = df.rename(columns={"region": "unique_id", "load_mw": "y"})[["unique_id", "ds", "y"]]
    starts = fold_starts(scale)

    t = time.time()
    best_cfg, val_table = select_conformal(feats, scale, log)
    timings["validation_s"] = time.time() - t
    log(f"selected conformal configuration: {config_name(*best_cfg)}")

    preds, fold_info, stats_s, lgbm_s = [], [], 0.0, 0.0
    fold_models = {}
    for k, fs in enumerate(starts):
        t = time.time()
        sp = stats_backtest_fold(series, fs, scale.fold_days, STATS_GROUPS)
        stats_s += time.time() - t
        t = time.time()
        raw, info = lgbm_fold_predictions(feats, fs, scale.fold_days, scale.cal_days, scale.seed)
        fold_models[k] = info.pop("models")
        sel = apply_cfg(raw, fs, best_cfg)
        static = conformalise(raw, fs, "SplitCP", scale.cal_days, "static")
        alt_method = ("CQR" if best_cfg[0] == "SplitCP" else "SplitCP", *best_cfg[1:])
        alt_groups = (*best_cfg[:3], 6 if best_cfg[3] == 24 else 24)
        alts = {cfg: apply_cfg(raw, fs, cfg) for cfg in (alt_method, alt_groups)}
        # Point model and raw (unconformalised) LightGBM quantiles, for comparison.
        test_raw = raw[raw["ds"] >= fs]
        rawq = pd.DataFrame(
            {
                "region": test_raw["region"],
                "ds": test_raw["ds"],
                "horizon": test_raw["horizon"],
                "y": test_raw["load_mw"],
                "q05": test_raw["p_lo"] * test_raw["level"],
                "q50": test_raw["p_med"] * test_raw["level"],
                "q95": test_raw["p_hi"] * test_raw["level"],
            }
        )
        lgbm_s += time.time() - t
        parts = [
            sp,
            sel.drop(columns="alpha_used").assign(model=SELECTED),
            static.drop(columns="alpha_used").assign(model=f"LightGBM+SplitCP static {scale.cal_days}d"),
            rawq.assign(model="LightGBM quantiles (no conformal)"),
        ]
        for cfg, a in alts.items():
            parts.append(a.drop(columns="alpha_used").assign(model=f"LightGBM+{config_name(*cfg)}"))
        for p in parts:
            preds.append(p.assign(fold=k))
        fold_info.append(
            {"fold": k, "start": str(fs.date()), **info, "mean_alpha_used": float(sel["alpha_used"].mean())}
        )
        log(f"  test fold {k} ({fs.date()}) done; stats {stats_s:.0f}s, lgbm {lgbm_s:.0f}s cumulative")
    timings["stats_models_s"] = stats_s
    timings["lightgbm_conformal_s"] = lgbm_s

    pred = pd.concat(preds, ignore_index=True)
    pred["horizon"] = pred["horizon"].astype(int)
    assert pred["horizon"].between(1, HORIZON).all()
    scales = mase_scales(df, starts)
    summary = summarise(pred, scales)
    order = list(dict.fromkeys(pred["model"]))
    summary["model"] = pd.Categorical(summary["model"], order)
    summary = summary.sort_values("mase").reset_index(drop=True)
    summary["model"] = summary["model"].astype(str)

    naive_names = [m.alias for m in STATS_GROUPS[0][1]]
    naive_best = summary[summary["model"].isin(naive_names)].sort_values("mase").iloc[0]
    summary["mase_vs_best_seasonal_naive_pct"] = (summary["mase"] / naive_best["mase"] - 1) * 100

    cov_rh = coverage_table(pred, ["model", "region", "horizon"])
    cov_r = coverage_table(pred, ["model", "region"])
    cov_f = coverage_table(pred, ["model", "fold"])
    per_fold = []
    for k in range(len(starts)):
        s = summarise(pred[pred["fold"] == k], scales[scales["fold"] == k])
        per_fold.append(s.assign(fold=k, start=str(starts[k].date())))
    per_fold = pd.concat(per_fold, ignore_index=True)
    per_region = []
    for r in sorted(pred["region"].unique()):
        s = summarise(pred[pred["region"] == r], scales[scales["region"] == r])
        per_region.append(s.assign(region=r))
    per_region = pd.concat(per_region, ignore_index=True)

    stress_tab = pd.DataFrame()
    if stress:
        t = time.time()
        stress_tab = stress_test(df, scale, starts, fold_models, best_cfg)
        timings["stress_s"] = time.time() - t
    timings["backtest_total_s"] = time.time() - t0
    return {
        "pred": pred,
        "summary": summary,
        "val_table": val_table,
        "best_cfg": best_cfg,
        "cov_region_horizon": cov_rh,
        "cov_region": cov_r,
        "cov_fold": cov_f,
        "per_fold": per_fold,
        "per_region": per_region,
        "fold_info": fold_info,
        "scales": scales,
        "stress": stress_tab,
        "timings": timings,
        "best_naive": str(naive_best["model"]),
        "fold_days": scale.fold_days,
        "cal_days": scale.cal_days,
        "scale": scale.name,
    }


def stress_test(df: pd.DataFrame, scale: Scale, starts, fold_models, best_cfg) -> pd.DataFrame:
    """Temperature scenarios: regenerate the same SYNTHETIC world with shifted temperature, forecast it.

    The generator's response under the scenario is the ground truth, so we can check whether the model's
    forecast peak and its conformal intervals hold up in weather it rarely saw.
    """
    rows = []
    for name, s0, s1, delta in STRESS_SCENARIOS:
        s0t, s1t = pd.Timestamp(s0), pd.Timestamp(s1)
        k = max(i for i, fs in enumerate(starts) if fs <= s0t)
        fs = starts[k]
        if s1t > fs + pd.Timedelta(days=scale.fold_days):
            continue
        stressed = generate(scale, temp_shift=(s0, s1, delta))
        for label, frame in (("base", df), ("scenario", stressed)):
            feats = build_features(frame)
            ev = feats[(feats["ds"] >= fs - pd.Timedelta(days=scale.cal_days)) & (feats["ds"] < s1t)]
            raw = cached_predictions(fold_models[k], ev.dropna(subset=["level", "y_rel"]))
            p = apply_cfg(raw, fs, best_cfg)
            p = p[(p["ds"] >= s0t) & (p["ds"] < s1t)]
            for region, g in p.groupby("region"):
                rows.append(
                    {
                        "scenario": name,
                        "delta_c": delta,
                        "window": f"{s0} to {s1}",
                        "world": label,
                        "region": region,
                        "actual_peak_mw": float(g["y"].max()),
                        "forecast_peak_mw": float(g.loc[g["y"].idxmax(), "q50"]),
                        "max_forecast_mw": float(g["q50"].max()),
                        "actual_energy_gwh": float(g["y"].sum() / 1000),
                        "forecast_energy_gwh": float(g["q50"].sum() / 1000),
                        "mape_pct": float(np.mean(np.abs(g["y"] - g["q50"]) / g["y"]) * 100),
                        "coverage90": float(np.mean((g["y"] >= g["q05"]) & (g["y"] <= g["q95"]))),
                        "n_hours": int(len(g)),
                    }
                )
    return pd.DataFrame(rows)


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")
