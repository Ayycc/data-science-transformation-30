"""Difference-in-means and CUPED (Deng et al., 2013) estimators.

The core routine ``stats_from_sums`` works on per-arm sufficient statistics so the same
formula serves one-shot analysis, daily sequential looks and vectorised Monte Carlo runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from expengine.ratio import relative_lift


@dataclass(frozen=True)
class EffectEstimate:
    metric: str
    method: str
    n_control: int
    n_treatment: int
    control_mean: float
    treatment_mean: float
    diff: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    rel_lift: float
    rel_ci_low: float
    rel_ci_high: float
    theta: float | None = None
    variance_reduction: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def arm_sums(y: np.ndarray, x: np.ndarray | None = None) -> dict[str, float]:
    """Sufficient statistics for one arm."""
    y = np.asarray(y, dtype=float)
    out = {"n": float(len(y)), "sy": float(y.sum()), "syy": float((y * y).sum())}
    if x is not None:
        x = np.asarray(x, dtype=float)
        out.update(sx=float(x.sum()), sxx=float((x * x).sum()), sxy=float((x * y).sum()))
    return out


def stats_from_sums(c: dict, t: dict, cuped: bool = False) -> dict[str, np.ndarray]:
    """Treatment-minus-control difference and its variance from per-arm sums.

    Inputs may be scalars or arrays of any (broadcastable) shape. With ``cuped=True`` a single
    pooled theta = cov(x, y) / var(x) is estimated from both arms and the adjusted metric
    y - theta * (x - mean_x) is compared. Returns arm means, the diff, var(diff) and theta.
    """
    n_c, n_t = np.asarray(c["n"], float), np.asarray(t["n"], float)
    my_c, my_t = c["sy"] / n_c, t["sy"] / n_t
    vy_c = (c["syy"] - n_c * my_c**2) / (n_c - 1)
    vy_t = (t["syy"] - n_t * my_t**2) / (n_t - 1)
    if not cuped:
        return {
            "mean_c": my_c,
            "mean_t": my_t,
            "diff": my_t - my_c,
            "var": vy_t / n_t + vy_c / n_c,
            "var_c": vy_c,
            "var_t": vy_t,
            "theta": np.zeros_like(my_c),
        }
    n = n_c + n_t
    sx, sxx, sxy, sy = c["sx"] + t["sx"], c["sxx"] + t["sxx"], c["sxy"] + t["sxy"], c["sy"] + t["sy"]
    mx = sx / n
    vx_pool = (sxx - n * mx**2) / (n - 1)
    cxy_pool = (sxy - n * mx * (sy / n)) / (n - 1)
    theta = np.divide(cxy_pool, vx_pool, out=np.zeros_like(np.asarray(cxy_pool, float)), where=vx_pool > 0)
    mx_c, mx_t = c["sx"] / n_c, t["sx"] / n_t
    vx_c = (c["sxx"] - n_c * mx_c**2) / (n_c - 1)
    vx_t = (t["sxx"] - n_t * mx_t**2) / (n_t - 1)
    cxy_c = (c["sxy"] - n_c * mx_c * my_c) / (n_c - 1)
    cxy_t = (t["sxy"] - n_t * mx_t * my_t) / (n_t - 1)
    adj_c = my_c - theta * (mx_c - mx)
    adj_t = my_t - theta * (mx_t - mx)
    va_c = np.maximum(vy_c - 2 * theta * cxy_c + theta**2 * vx_c, 0.0)
    va_t = np.maximum(vy_t - 2 * theta * cxy_t + theta**2 * vx_t, 0.0)
    return {
        "mean_c": adj_c,
        "mean_t": adj_t,
        "diff": adj_t - adj_c,
        "var": va_t / n_t + va_c / n_c,
        "var_c": va_c,
        "var_t": va_t,
        "theta": theta,
    }


def _build(metric: str, method: str, s: dict, n_c: int, n_t: int, alpha: float, raw_means=None, vr=None):
    se = float(np.sqrt(s["var"]))
    diff = float(s["diff"])
    z = stats.norm.ppf(1 - alpha / 2)
    if se > 0:
        p = float(2 * stats.norm.sf(abs(diff) / se))
    else:
        p = 1.0 if diff == 0 else 0.0
    mc, mt = raw_means if raw_means is not None else (float(s["mean_c"]), float(s["mean_t"]))
    rel, rlo, rhi = relative_lift(
        float(s["mean_t"]), float(s["var_t"]) / n_t, float(s["mean_c"]), float(s["var_c"]) / n_c, alpha
    )
    return EffectEstimate(
        metric=metric,
        method=method,
        n_control=int(n_c),
        n_treatment=int(n_t),
        control_mean=float(mc),
        treatment_mean=float(mt),
        diff=diff,
        se=se,
        ci_low=diff - z * se,
        ci_high=diff + z * se,
        p_value=p,
        rel_lift=rel,
        rel_ci_low=rlo,
        rel_ci_high=rhi,
        theta=float(s["theta"]) if method != "naive" else None,
        variance_reduction=vr,
    )


def diff_in_means(y: np.ndarray, treat: np.ndarray, alpha: float = 0.05, metric: str = "y") -> EffectEstimate:
    """Welch-style difference in means with a normal-approximation CI."""
    y, treat = np.asarray(y, float), np.asarray(treat, bool)
    s = stats_from_sums(arm_sums(y[~treat]), arm_sums(y[treat]))
    return _build(metric, "naive", s, int((~treat).sum()), int(treat.sum()), alpha)


def cuped_theta(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Pooled CUPED coefficient(s): Cov(X)^-1 Cov(X, y). ``x`` may be 1-D or (n, k)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    sxx = xc.T @ xc
    sxy = xc.T @ yc
    return np.linalg.lstsq(sxx, sxy, rcond=None)[0]


def cuped_adjust(y: np.ndarray, x: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
    """Return y - (x - mean(x)) @ theta (multi-covariate CUPED)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    if theta is None:
        theta = cuped_theta(y, x)
    return y - (x - x.mean(axis=0)) @ np.atleast_1d(theta)


def cuped_estimate(
    y: np.ndarray, x: np.ndarray, treat: np.ndarray, alpha: float = 0.05, metric: str = "y"
) -> EffectEstimate:
    """CUPED estimate of the treatment effect using pre-period covariate(s) ``x``.

    ``variance_reduction`` is 1 - var(diff_cuped) / var(diff_naive), estimated in-sample.
    Means reported are the raw arm means; ``diff`` is the adjusted difference.
    """
    y, treat = np.asarray(y, float), np.asarray(treat, bool)
    x = np.asarray(x, float)
    naive = stats_from_sums(arm_sums(y[~treat]), arm_sums(y[treat]))
    if x.ndim == 1:
        s = stats_from_sums(arm_sums(y[~treat], x[~treat]), arm_sums(y[treat], x[treat]), cuped=True)
    else:
        theta = cuped_theta(y, x)
        ya = cuped_adjust(y, x, theta)
        s = stats_from_sums(arm_sums(ya[~treat]), arm_sums(ya[treat]))
        s["theta"] = float(theta[0])
    vr = float(1 - s["var"] / naive["var"]) if naive["var"] > 0 else 0.0
    raw = (float(naive["mean_c"]), float(naive["mean_t"]))
    return _build(metric, "cuped", s, int((~treat).sum()), int(treat.sum()), alpha, raw_means=raw, vr=vr)


def regression_adjusted(y: np.ndarray, x: np.ndarray, treat: np.ndarray) -> dict[str, float]:
    """ANCOVA cross-check with statsmodels OLS (HC1 robust SE): y ~ 1 + treat + (x - mean x)."""
    import statsmodels.api as sm

    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    design = np.column_stack([np.asarray(treat, float), x - x.mean(axis=0)])
    fit = sm.OLS(y, sm.add_constant(design)).fit(cov_type="HC1")
    return {"diff": float(fit.params[1]), "se": float(fit.bse[1]), "p_value": float(fit.pvalues[1])}
