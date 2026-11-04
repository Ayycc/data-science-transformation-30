"""Differentially private aggregate releases (counts, means, histograms) and accuracy-vs-epsilon analysis.

Neighbouring datasets differ by adding or removing one person (unbounded DP). Under that definition a full
contingency table over a *public* domain has L1 and L2 sensitivity 1, and a sum of values clipped to
[lo, hi] has sensitivity max(|lo|, |hi|). Official release values are drawn with diffprivlib's mechanisms;
the Monte-Carlo accuracy curves draw the identical distributions with vectorised NumPy (a test checks the
noise scales match).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from diffprivlib.mechanisms import GaussianAnalytic, Laplace

from .accounting import PrivacyAccountant, dp_to_zcdp_rho, zcdp_to_dp
from .data import CATEGORICAL, NUMERIC_BOUNDS, RARE_GROUP
from .encoding import BINS


def laplace_scale(sensitivity: float, epsilon: float) -> float:
    return sensitivity / epsilon


def zcdp_gaussian_sigma(sensitivity: float, rho: float) -> float:
    return sensitivity / math.sqrt(2 * rho)


def contingency(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Exact counts over the full public domain (zero cells included, so the table shape leaks nothing)."""
    levels = [CATEGORICAL[c] if c in CATEGORICAL else list(range(len(BINS[c]) - 1)) for c in cols]
    keys = []
    for c in cols:
        if c in BINS:
            keys.append(pd.Series(np.clip(np.searchsorted(BINS[c], df[c], side="right") - 1, 0, len(BINS[c]) - 2), index=df.index))
        else:
            keys.append(df[c])
    counts = pd.Series(1, index=df.index).groupby(keys).sum()
    full = pd.MultiIndex.from_product(levels, names=cols) if len(cols) > 1 else pd.Index(levels[0], name=cols[0])
    return counts.reindex(full, fill_value=0).astype(float)


def _diffprivlib_noise(values: np.ndarray, mech) -> np.ndarray:
    return np.array([mech.randomise(float(v)) for v in values])


def dp_counts(df: pd.DataFrame, cols: list[str], epsilon: float, seed: int, delta: float = 0.0) -> tuple[pd.Series, pd.Series]:
    """(noisy, exact) contingency table: Laplace if delta == 0, else analytic Gaussian (diffprivlib)."""
    exact = contingency(df, cols)
    if delta == 0:
        mech = Laplace(epsilon=epsilon, sensitivity=1.0, random_state=seed)
    else:
        mech = GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=1.0, random_state=seed)
    noisy = pd.Series(_diffprivlib_noise(exact.to_numpy(), mech), index=exact.index)
    return noisy, exact


def postprocess_counts(noisy: pd.Series) -> pd.Series:
    """Round and clip at zero (post-processing costs no privacy)."""
    return noisy.round().clip(lower=0)


@dataclass
class ReleasePlan:
    """Budget split of the aggregate-table release (fractions of ``epsilon``)."""

    epsilon: float = 1.0
    race_sex_income: float = 0.4
    age_histogram: float = 0.2
    hours_sum_by_race: float = 0.2
    education_income: float = 0.2


def official_release(df: pd.DataFrame, accountant: PrivacyAccountant, plan: ReleasePlan, seed: int) -> dict:
    """Publish four DP tables, spending through ``accountant`` *before* any noise is drawn."""
    sub = accountant.sub_budget("aggregate tables", plan.epsilon)
    out: dict = {"tables": {}}

    e1 = plan.epsilon * plan.race_sex_income
    sub.spend("T1 counts race x sex x income (Laplace)", e1)
    noisy, exact = dp_counts(df, ["race", "sex", "income"], e1, seed)
    t1 = postprocess_counts(noisy)
    out["tables"]["T1_race_sex_income"] = _table_error(t1, exact, rare_mask=exact.index.get_level_values("race") == RARE_GROUP[1])

    e2 = plan.epsilon * plan.age_histogram
    sub.spend("T2 age histogram, 5-year bins (Laplace)", e2)
    noisy, exact = dp_counts(df, ["age"], e2, seed + 1)
    t2 = postprocess_counts(noisy)
    out["tables"]["T2_age_histogram"] = _table_error(t2, exact) | {"tvd": float(0.5 * np.abs(t2 / t2.sum() - exact / exact.sum()).sum())}

    # T3: mean hours by race = noisy clipped sum (this spend) / T1 counts marginalised over sex and income (post-processing)
    e3 = plan.epsilon * plan.hours_sum_by_race
    sub.spend("T3 sum of hours by race, clipped [1, 99] (Laplace; parallel over races)", e3)
    lo, hi = NUMERIC_BOUNDS["hours_per_week"]
    mech = Laplace(epsilon=e3, sensitivity=max(abs(lo), abs(hi)), random_state=seed + 2)
    sums = df.groupby("race")["hours_per_week"].sum().reindex(CATEGORICAL["race"], fill_value=0).astype(float)
    noisy_sums = pd.Series(_diffprivlib_noise(sums.to_numpy(), mech), index=sums.index)
    counts = t1.groupby(level="race").sum().reindex(CATEGORICAL["race"]).clip(lower=1)
    means_dp = (noisy_sums / counts).clip(lo, hi)
    means = df.groupby("race")["hours_per_week"].mean().reindex(CATEGORICAL["race"])
    out["tables"]["T3_mean_hours_by_race"] = {
        "dp": means_dp.round(3).to_dict(),
        "exact": means.round(3).to_dict(),
        "abs_error": (means_dp - means).abs().round(3).to_dict(),
    }

    e4 = plan.epsilon * plan.education_income
    sub.spend("T4 counts education x income (Laplace)", e4)
    noisy, exact = dp_counts(df, ["education", "income"], e4, seed + 3)
    t4 = postprocess_counts(noisy)
    out["tables"]["T4_education_income"] = _table_error(t4, exact)
    out["budget"] = sub.to_dict()
    return out


def _table_error(noisy: pd.Series, exact: pd.Series, rare_mask: np.ndarray | None = None) -> dict:
    err = (noisy - exact).abs()
    nz = exact > 0
    res = {
        "cells": int(len(exact)),
        "mean_abs_error": float(err.mean()),
        "max_abs_error": float(err.max()),
        "median_rel_error_nonzero_cells": float((err[nz] / exact[nz]).median()),
    }
    if rare_mask is not None:
        rm = np.asarray(rare_mask) & nz.to_numpy()
        om = ~np.asarray(rare_mask) & nz.to_numpy()
        res["rare_group_mean_rel_error"] = float((err[rm] / exact[rm]).mean())
        res["other_groups_mean_rel_error"] = float((err[om] / exact[om]).mean())
        res["rare_group_cells"] = {str(k): {"exact": float(exact[k]), "released": float(noisy[k])} for k in exact.index[np.asarray(rare_mask)]}
    return res


def accuracy_curves(df: pd.DataFrame, epsilons: list[float], trials: int, seed: int, delta: float = 1e-6) -> dict:
    """Error of each released statistic as a function of epsilon (Monte Carlo, same noise law as the release)."""
    rng = np.random.default_rng(seed)
    t1 = contingency(df, ["race", "sex", "income"])
    rare = (t1.index.get_level_values("race") == RARE_GROUP[1]) & (t1.to_numpy() > 0)
    white = (t1.index.get_level_values("race") == "White") & (t1.to_numpy() > 0)
    age = contingency(df, ["age"]).to_numpy()
    lo, hi = NUMERIC_BOUNDS["hours_per_week"]
    by_race = df.groupby("race")["hours_per_week"].agg(["sum", "count", "mean"]).reindex(CATEGORICAL["race"])
    t1v = t1.to_numpy()
    out = {k: [] for k in ["epsilon", "count_rel_err_white", "count_rel_err_rare", "count_rel_err_white_gauss", "count_rel_err_rare_gauss"]}
    out |= {k: [] for k in ["age_hist_tvd", "mean_hours_abs_err_white", "mean_hours_abs_err_rare", "overall_count_rel_err"]}
    n = len(df)
    for eps in epsilons:
        b = laplace_scale(1.0, eps)
        noise = rng.laplace(0, b, size=(trials, len(t1v)))
        rel = np.abs(noise) / np.maximum(t1v, 1)
        sig = zcdp_gaussian_sigma(1.0, dp_to_zcdp_rho(eps, delta))
        relg = np.abs(rng.normal(0, sig, size=(trials, len(t1v)))) / np.maximum(t1v, 1)
        out["epsilon"].append(eps)
        out["count_rel_err_white"].append(float(rel[:, white].mean()))
        out["count_rel_err_rare"].append(float(rel[:, rare].mean()))
        out["count_rel_err_white_gauss"].append(float(relg[:, white].mean()))
        out["count_rel_err_rare_gauss"].append(float(relg[:, rare].mean()))
        out["overall_count_rel_err"].append(float(np.mean(np.abs(rng.laplace(0, b, trials)) / n)))
        h = np.clip(np.round(age[None, :] + rng.laplace(0, b, size=(trials, len(age)))), 0, None)
        h = h / np.maximum(h.sum(1, keepdims=True), 1)
        out["age_hist_tvd"].append(float((0.5 * np.abs(h - age / age.sum()).sum(1)).mean()))
        sum_noise = rng.laplace(0, laplace_scale(hi, eps / 2), size=(trials, len(by_race)))
        cnt_noise = rng.laplace(0, laplace_scale(1.0, eps / 2), size=(trials, len(by_race)))
        m = np.clip((by_race["sum"].to_numpy() + sum_noise) / np.maximum(by_race["count"].to_numpy() + cnt_noise, 1), lo, hi)
        err = np.abs(m - by_race["mean"].to_numpy())
        out["mean_hours_abs_err_white"].append(float(err[:, CATEGORICAL["race"].index("White")].mean()))
        out["mean_hours_abs_err_rare"].append(float(err[:, CATEGORICAL["race"].index(RARE_GROUP[1])].mean()))
    out["rare_group_cell_counts"] = [float(v) for v in t1v[rare]]
    out["white_cell_counts"] = [float(v) for v in t1v[white]]
    return out


def composition_comparison(epsilon: float, delta: float, ks: list[int]) -> dict:
    """Noise std per count query when a fixed (epsilon, delta) budget is split over k queries.

    Laplace + basic composition: each query gets epsilon/k, std = sqrt(2) * k / epsilon.
    Gaussian + zCDP: total rho from (epsilon, delta), each query gets rho/k, std = sqrt(k / (2 rho)).
    """
    rho = dp_to_zcdp_rho(epsilon, delta)
    lap = [math.sqrt(2) * k / epsilon for k in ks]
    gau = [zcdp_gaussian_sigma(1.0, rho / k) for k in ks]
    crossover = next((k for k, a, g in zip(ks, lap, gau) if g < a), None)
    return {
        "epsilon": epsilon,
        "delta": delta,
        "rho": rho,
        "rho_back_to_eps": zcdp_to_dp(rho, delta),
        "k": ks,
        "laplace_std": lap,
        "gaussian_zcdp_std": gau,
        "crossover_k": crossover,
    }
