"""Bayesian A/B analysis: Beta-Binomial for conversions, Normal for continuous effects."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def beta_binomial(
    conv_c: int,
    n_c: int,
    conv_t: int,
    n_t: int,
    prior: tuple[float, float] = (1.0, 1.0),
    draws: int = 200_000,
    seed: int = 0,
) -> dict[str, float]:
    """Posterior summary for two conversion rates with independent Beta priors.

    expected_loss_* is the posterior expected shortfall (in absolute rate) if you choose that
    arm and it is in fact worse: E[max(p_other - p_chosen, 0)].
    """
    a, b = prior
    rng = np.random.default_rng(seed)
    pc = rng.beta(a + conv_c, b + n_c - conv_c, draws)
    pt = rng.beta(a + conv_t, b + n_t - conv_t, draws)
    lift = pt / pc - 1
    return {
        "post_mean_c": float((a + conv_c) / (a + b + n_c)),
        "post_mean_t": float((a + conv_t) / (a + b + n_t)),
        "prob_t_better": float((pt > pc).mean()),
        "expected_loss_t": float(np.maximum(pc - pt, 0).mean()),
        "expected_loss_c": float(np.maximum(pt - pc, 0).mean()),
        "rel_lift_median": float(np.median(lift)),
        "rel_lift_ci_low": float(np.quantile(lift, 0.025)),
        "rel_lift_ci_high": float(np.quantile(lift, 0.975)),
    }


def normal_posterior(diff: float, se: float, prior_mean: float = 0.0, prior_sd: float = math.inf) -> dict[str, float]:
    """Normal-Normal posterior for a treatment effect given an estimate and its SE.

    With ``prior_sd=inf`` (flat prior) the posterior is N(diff, se^2). A finite prior shrinks
    noisy estimates toward ``prior_mean`` (e.g. the historical distribution of effects).
    """
    if se <= 0:
        raise ValueError("se must be positive")
    if math.isinf(prior_sd):
        m, s = diff, se
    else:
        w = 1 / se**2 + 1 / prior_sd**2
        m = (diff / se**2 + prior_mean / prior_sd**2) / w
        s = math.sqrt(1 / w)
    z = m / s
    loss_t = s * stats.norm.pdf(z) - m * stats.norm.cdf(-z)  # E[max(-delta, 0)]
    loss_c = s * stats.norm.pdf(z) + m * stats.norm.cdf(z)  # E[max(delta, 0)]
    return {
        "post_mean": float(m),
        "post_sd": float(s),
        "prob_positive": float(stats.norm.cdf(z)),
        "expected_loss_t": float(loss_t),
        "expected_loss_c": float(loss_c),
        "ci_low": float(m - 1.959963984540054 * s),
        "ci_high": float(m + 1.959963984540054 * s),
    }


def bayes_decision(
    prob_better: float, expected_loss_t: float, loss_tolerance: float, prob_threshold: float = 0.95
) -> str:
    """Decision rule: ship when the treatment is probably better *and* the downside is small."""
    if prob_better >= prob_threshold and expected_loss_t <= loss_tolerance:
        return "ship"
    if prob_better <= 1 - prob_threshold:
        return "do not ship"
    return "keep running / inconclusive"
