"""Mixture sequential probability ratio test (mSPRT) with always-valid p-values and CIs.

Following Johari, Pekelis and Walsh (2017, "Always valid inference"): with a normal mixing
distribution N(0, tau^2) over the true difference theta, and an estimate theta_hat with
variance V at each look, the mixture likelihood ratio against H0: theta = 0 is

    Lambda = sqrt(V / (V + tau^2)) * exp(tau^2 * theta_hat^2 / (2 V (V + tau^2)))

and p_n = min(p_{n-1}, 1 / Lambda_n) is valid however often (and whenever) you peek.
All functions broadcast, with looks on the last axis.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def msprt_lambda(diff: np.ndarray, var: np.ndarray, tau2: float) -> np.ndarray:
    """Mixture likelihood ratio for H0: difference = 0 (log-safe, capped to avoid overflow)."""
    diff = np.asarray(diff, float)
    var = np.asarray(var, float)
    if tau2 <= 0:
        raise ValueError("tau2 must be positive")
    log_lam = 0.5 * np.log(var / (var + tau2)) + tau2 * diff**2 / (2 * var * (var + tau2))
    return np.exp(np.minimum(log_lam, 700.0))


def msprt_pvalues(diff: np.ndarray, var: np.ndarray, tau2: float) -> np.ndarray:
    """Always-valid p-values: running minimum of min(1, 1 / Lambda) along the last axis."""
    p = np.minimum(1.0, 1.0 / msprt_lambda(diff, var, tau2))
    return np.minimum.accumulate(p, axis=-1)


def always_valid_ci(
    diff: np.ndarray, var: np.ndarray, tau2: float, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Always-valid (1 - alpha) confidence sequence, intersected over looks."""
    diff = np.asarray(diff, float)
    var = np.asarray(var, float)
    half = np.sqrt(var * (var + tau2) / tau2 * (np.log((var + tau2) / var) - 2 * np.log(alpha)))
    lo = np.maximum.accumulate(diff - half, axis=-1)
    hi = np.minimum.accumulate(diff + half, axis=-1)
    return lo, hi


def naive_pvalues(diff: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Fixed-horizon two-sided z-test p-value at each look (invalid if you peek)."""
    return 2 * stats.norm.sf(np.abs(np.asarray(diff, float)) / np.sqrt(np.asarray(var, float)))


def msprt_sequence(diff: np.ndarray, var: np.ndarray, tau2: float, alpha: float = 0.05) -> dict:
    """Full sequential trace for one experiment: p-values, CIs and the first rejecting look."""
    diff = np.asarray(diff, float)
    var = np.asarray(var, float)
    p = msprt_pvalues(diff, var, tau2)
    lo, hi = always_valid_ci(diff, var, tau2, alpha)
    rejected = np.flatnonzero(p < alpha)
    naive = naive_pvalues(diff, var)
    naive_rej = np.flatnonzero(naive < alpha)
    return {
        "p_always_valid": p,
        "ci_low": lo,
        "ci_high": hi,
        "p_naive": naive,
        "first_reject_look": int(rejected[0]) if len(rejected) else None,
        "first_naive_reject_look": int(naive_rej[0]) if len(naive_rej) else None,
    }


def tau2_from_effect(effect_abs: float) -> float:
    """Mixing variance matched to the effect size you expect (or care about) to detect."""
    if effect_abs <= 0:
        raise ValueError("effect_abs must be positive")
    return float(effect_abs**2)
