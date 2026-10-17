"""Power and sample-size calculations for two-arm tests (normal approximation)."""

from __future__ import annotations

import math

from scipy import stats


def _z(alpha: float, power: float, two_sided: bool = True) -> tuple[float, float]:
    za = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    zb = stats.norm.ppf(power)
    return float(za), float(zb)


def n_per_arm_means(
    sigma: float,
    mde_abs: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    variance_reduction: float = 0.0,
) -> int:
    """Users needed in the control arm to detect an absolute difference ``mde_abs``.

    ``ratio`` is n_treatment / n_control. ``variance_reduction`` (e.g. from CUPED, 1 - rho^2
    remaining) scales the variance by ``1 - variance_reduction``.
    """
    if sigma <= 0 or mde_abs <= 0:
        raise ValueError("sigma and mde_abs must be positive")
    if not 0 <= variance_reduction < 1:
        raise ValueError("variance_reduction must be in [0, 1)")
    za, zb = _z(alpha, power)
    var = sigma**2 * (1 - variance_reduction)
    n = (za + zb) ** 2 * var * (1 + 1 / ratio) / mde_abs**2
    return int(math.ceil(n))


def n_per_arm_proportions(
    p_baseline: float,
    mde_rel: float,
    alpha: float = 0.05,
    power: float = 0.8,
    variance_reduction: float = 0.0,
) -> int:
    """Users per arm to detect a relative lift ``mde_rel`` on a conversion rate (equal arms)."""
    if not 0 < p_baseline < 1:
        raise ValueError("p_baseline must be in (0, 1)")
    p2 = p_baseline * (1 + mde_rel)
    if not 0 < p2 < 1:
        raise ValueError("treatment rate outside (0, 1)")
    za, zb = _z(alpha, power)
    var = (p_baseline * (1 - p_baseline) + p2 * (1 - p2)) * (1 - variance_reduction)
    return int(math.ceil((za + zb) ** 2 * var / (p2 - p_baseline) ** 2))


def mde_means(sigma: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable absolute difference with ``n_per_arm`` users in each arm."""
    za, zb = _z(alpha, power)
    return float((za + zb) * sigma * math.sqrt(2 / n_per_arm))


def power_means(sigma: float, mde_abs: float, n_per_arm: int, alpha: float = 0.05) -> float:
    """Power of a two-sided z-test for an absolute difference ``mde_abs``."""
    za = stats.norm.ppf(1 - alpha / 2)
    se = sigma * math.sqrt(2 / n_per_arm)
    shift = mde_abs / se
    return float(stats.norm.cdf(shift - za) + stats.norm.cdf(-shift - za))
