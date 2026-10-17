"""Delta-method variance for ratio metrics and relative lifts."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def ratio_mean_var(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """Ratio of sums sum(num) / sum(den) and its delta-method variance.

    Used when the analysis unit (e.g. a session) differs from the randomisation unit (a
    user): per-user numerators and denominators are correlated, so a naive per-session
    variance would be too small.
    """
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    n = len(num)
    if n < 2 or den.sum() <= 0:
        raise ValueError("need >= 2 units and a positive denominator")
    mx, my = num.mean(), den.mean()
    vx, vy = num.var(ddof=1), den.var(ddof=1)
    cxy = np.cov(num, den, ddof=1)[0, 1]
    r = mx / my
    var = (vx / my**2 - 2 * mx * cxy / my**3 + mx**2 * vy / my**4) / n
    return float(r), float(max(var, 0.0))


def ratio_metric_test(num: np.ndarray, den: np.ndarray, treat: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """Two-arm test of a ratio metric (e.g. revenue per session) with delta-method SEs."""
    treat = np.asarray(treat, bool)
    num, den = np.asarray(num, float), np.asarray(den, float)
    rc, vc = ratio_mean_var(num[~treat], den[~treat])
    rt, vt = ratio_mean_var(num[treat], den[treat])
    diff = rt - rc
    se = math.sqrt(vc + vt)
    z = stats.norm.ppf(1 - alpha / 2)
    rel, rlo, rhi = relative_lift(rt, vt, rc, vc, alpha)
    return {
        "control_ratio": rc,
        "treatment_ratio": rt,
        "diff": diff,
        "se": se,
        "ci_low": diff - z * se,
        "ci_high": diff + z * se,
        "p_value": float(2 * stats.norm.sf(abs(diff) / se)) if se > 0 else 1.0,
        "rel_lift": rel,
        "rel_ci_low": rlo,
        "rel_ci_high": rhi,
    }


def relative_lift(
    mean_t: float, var_mean_t: float, mean_c: float, var_mean_c: float, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Relative lift mean_t / mean_c - 1 with a delta-method CI (independent arms)."""
    if mean_c == 0:
        return float("nan"), float("nan"), float("nan")
    rel = mean_t / mean_c - 1
    var = var_mean_t / mean_c**2 + mean_t**2 * var_mean_c / mean_c**4
    z = stats.norm.ppf(1 - alpha / 2)
    se = math.sqrt(max(var, 0.0))
    return float(rel), float(rel - z * se), float(rel + z * se)
