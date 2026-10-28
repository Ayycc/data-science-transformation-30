"""Budget optimiser over posterior draws: expected and risk-adjusted allocations under budget and channel bounds.

Decision variable: weekly spend per channel, in $k per million people, applied in every region (so regional
spend stays proportional to population). The response is the steady-state weekly media contribution: with
normalised adstock, a constant weekly spend s settles at an adstocked level of s, so the contribution is
sum_r pop_r * y_scale * beta[r,c] * logistic(s_c / scale_c; lam_c).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize

from .config import HDI_PROB, SEED


@dataclass
class ResponseModel:
    beta: np.ndarray  # (S, R, C) in scaled per-capita units at full saturation
    lam: np.ndarray  # (S, C)
    population: np.ndarray  # (R,)
    y_scale: float
    channel_scale: np.ndarray  # (C,)

    def __post_init__(self) -> None:
        # Pre-aggregate over regions: (S, C) population-weighted ceiling in $k per week.
        self._ceiling = np.einsum("src,r->sc", self.beta, self.population) * self.y_scale

    def draws(self, s: np.ndarray) -> np.ndarray:
        """Weekly media contribution per draw, (S,) in $k, for per-capita channel spend s (C,)."""
        x = np.asarray(s, dtype=float) / self.channel_scale
        e = np.exp(-self.lam * x[None, :])
        return (self._ceiling * (1 - e) / (1 + e)).sum(axis=1)

    def channel_draws(self, s: np.ndarray) -> np.ndarray:
        x = np.asarray(s, dtype=float) / self.channel_scale
        e = np.exp(-self.lam * x[None, :])
        return self._ceiling * (1 - e) / (1 + e)

    def grad_draws(self, s: np.ndarray) -> np.ndarray:
        """d contribution / d s_c per draw, (S, C)."""
        x = np.asarray(s, dtype=float) / self.channel_scale
        e = np.exp(-self.lam * x[None, :])
        dsat = 2 * self.lam * e / (1 + e) ** 2 / self.channel_scale
        return self._ceiling * dsat


def objective(rm: ResponseModel, s: np.ndarray, risk_aversion: float) -> tuple[float, np.ndarray]:
    """Negative (mean - k * sd) of total contribution, with gradient."""
    d = rm.draws(s)
    g = rm.grad_draws(s)
    mean, sd = d.mean(), d.std()
    gmean = g.mean(axis=0)
    val = mean
    grad = gmean
    if risk_aversion > 0:
        gsd = ((d - mean)[:, None] * (g - gmean)).mean(axis=0) / max(sd, 1e-12)
        val = mean - risk_aversion * sd
        grad = gmean - risk_aversion * gsd
    return -val, -grad


def optimise(
    rm: ResponseModel, s0: np.ndarray, lower: np.ndarray, upper: np.ndarray, risk_aversion: float = 0.0, n_starts: int = 8, seed: int = SEED
) -> np.ndarray:
    """Maximise mean (or mean - k*sd) weekly contribution subject to sum(s) = sum(s0) and lower <= s <= upper."""
    budget = float(np.sum(s0))
    rng = np.random.default_rng(seed)
    starts = [np.asarray(s0, dtype=float)]
    for _ in range(n_starts - 1):
        w = rng.dirichlet(np.ones(len(s0)))
        starts.append(np.clip(lower + w * (budget - lower.sum()), lower, upper))
    cons = [{"type": "eq", "fun": lambda s: s.sum() - budget, "jac": lambda s: np.ones_like(s)}]
    best, best_val = None, np.inf
    for x0 in starts:
        res = minimize(
            lambda s: objective(rm, s, risk_aversion),
            x0,
            jac=True,
            method="SLSQP",
            bounds=list(zip(lower, upper, strict=True)),
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if res.success and res.fun < best_val and abs(res.x.sum() - budget) < 1e-6 * max(budget, 1):
            best, best_val = res.x, res.fun
    if best is None:
        raise RuntimeError("budget optimisation failed from every start")
    return best


def linear_allocation(scores: np.ndarray, s0: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Allocate the same budget to maximise sum(score_c * s_c), e.g. trusting last-click ROAS at face value."""
    res = linprog(-np.asarray(scores), A_eq=np.ones((1, len(s0))), b_eq=[float(np.sum(s0))], bounds=list(zip(lower, upper, strict=True)), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def summarise_lift(rm: ResponseModel, s_new: np.ndarray, s0: np.ndarray, weeks: int = 52) -> dict:
    """Posterior lift of a new allocation vs current, over `weeks` weeks, in $k."""
    import arviz as az

    lift = (rm.draws(s_new) - rm.draws(s0)) * weeks
    base = rm.draws(s0) * weeks
    lo, hi = az.hdi(lift, hdi_prob=HDI_PROB)
    plo, phi = az.hdi(lift / base, hdi_prob=HDI_PROB)
    return {
        "lift_mean_k": float(lift.mean()),
        "lift_hdi_low_k": float(lo),
        "lift_hdi_high_k": float(hi),
        "lift_pct_of_media_contribution_mean": float((lift / base).mean()),
        "lift_pct_hdi_low": float(plo),
        "lift_pct_hdi_high": float(phi),
        "prob_lift_positive": float((lift > 0).mean()),
        "lift_p05_k": float(np.quantile(lift, 0.05)),
        "contribution_mean_k": float((rm.draws(s_new) * weeks).mean()),
        "contribution_sd_k": float((rm.draws(s_new) * weeks).std()),
    }
