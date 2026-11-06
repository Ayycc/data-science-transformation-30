"""Pricing policies and the guardrail layer.

Every policy picks from a guardrail-feasible candidate set built by :class:`Guardrails`, and the
simulator re-checks the posted price, so a policy cannot post an infeasible rate.

Policies
  * ``static``        the incumbent seasonal rate table (no inventory control).
  * ``greedy_naive``  semi-log demand fitted by naive OLS on the logs + DP inventory control.
  * ``greedy_iv``     the same with the 2SLS (instrumented) elasticity.
  * ``eps_greedy``    online ridge demand model per booking window, epsilon_t = min(0.3, t^-1/3).
  * ``linucb``        optimistic (UCB) demand estimate, alpha * sigma_hat * ||phi||_{A^-1}.
  * ``thompson``      Bayesian linear regression, one posterior draw per decision.
  * ``linucb_no_rm``  LinUCB with the continuation value set to zero (myopic, ablation).
  * ``oracle``        knows the true expected demand curve (not the hidden shock) and the true DP.
The bandits learn demand from scratch (cold start) and use the RM layer's continuation value
V_{j+1} (from the instrumented model) to price inventory: price = argmax E[p*min(N,R) + V(R-min(N,R))].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rm import step_values

GRID = np.round(np.arange(0.6, 1.81, 0.1), 2)  # price multipliers of the static rate


@dataclass(frozen=True)
class Guardrails:
    floor: float = 79.0
    ceiling: float = 449.0
    max_change: float = 0.20  # max relative change vs the previous night's price in the same booking window

    def bounds(self, prev: float | None) -> tuple[float, float]:
        lo, hi = self.floor, self.ceiling
        if prev is not None:
            lo = max(lo, prev * (1 - self.max_change))
            hi = min(hi, prev * (1 + self.max_change))
        return lo, hi

    def candidates(self, static: float, prev: float | None) -> np.ndarray:
        lo, hi = self.bounds(prev)
        c = np.clip(static * GRID, lo, hi)
        return np.unique(np.round(c, 6))

    def is_ok(self, price: float, prev: float | None, tol: float = 1e-6) -> bool:
        lo, hi = self.bounds(prev)
        return lo - tol <= price <= hi + tol


def features(ctx, m: np.ndarray) -> np.ndarray:
    """phi(x, m) = x (x) [1, m, m^2]; rows = candidate multipliers."""
    x = ctx.features()
    m = np.atleast_1d(m)
    basis = np.column_stack([np.ones_like(m), m, m**2])
    return (basis[:, None, :] * x[None, :, None]).reshape(len(m), -1)


class Policy:
    name = "policy"
    uses_rm = True

    def choose(self, ctx, cands: np.ndarray, R: int, v_next: np.ndarray, rng: np.random.Generator) -> float:
        raise NotImplementedError

    def update(self, ctx, price: float, demand: int) -> None:
        pass


class StaticPolicy(Policy):
    name = "static"
    uses_rm = False

    def choose(self, ctx, cands, R, v_next, rng):
        return float(cands[np.argmin(np.abs(cands - ctx.static))])


class ModelPolicy(Policy):
    """Greedy DP pricing with a fixed demand model (``predict(ctx, prices)``)."""

    def __init__(self, model, name: str):
        self.model, self.name = model, name

    def choose(self, ctx, cands, R, v_next, rng):
        return float(cands[np.argmax(step_values(cands, self.model.predict(ctx, cands), R, v_next))])


class OraclePolicy(Policy):
    name = "oracle"

    def __init__(self, cfg):
        self.cfg = cfg

    def choose(self, ctx, cands, R, v_next, rng):
        from .simulator import true_expected_demand

        lam = true_expected_demand(ctx, cands, self.cfg, ctx.u_weights)
        return float(cands[np.argmax(step_values(cands, lam, R, v_next))])


class _Ridge:
    """Online ridge regression with Sherman-Morrison updates and an in-sample noise estimate."""

    def __init__(self, dim: int, lam: float = 1.0, sigma0: float = 10.0):
        self.Ainv = np.eye(dim) / lam
        self.b = np.zeros(dim)
        self.theta = np.zeros(dim)
        self.n = 0
        self.yy = 0.0
        self.dim = dim
        self.sigma0 = sigma0

    @property
    def sigma(self) -> float:
        if self.n < 2 * self.dim:
            return self.sigma0
        sse = max(self.yy - float(self.theta @ self.b), 0.0)  # y'y - theta'X'y (ridge residual sum of squares, approx.)
        return float(np.sqrt(sse / (self.n - self.dim)))

    def update(self, phi: np.ndarray, y: float) -> None:
        Av = self.Ainv @ phi
        self.Ainv -= np.outer(Av, Av) / (1.0 + phi @ Av)
        self.b += y * phi
        self.yy += y * y
        self.theta = self.Ainv @ self.b
        self.n += 1


class LinearBandit(Policy):
    """Shared machinery: one ridge model of log(demand + 0.5) per booking window over phi(x, m).

    The log link makes the model multiplicative in season, events and lead time, like real demand.
    ``warm_start`` seeds the models with the randomised price-test cells of the logs only, because
    the other logged prices were set on the hidden demand shock and would bias the slope.
    """

    def __init__(self, name: str, n_buckets: int = 3, use_rm: bool = True, lam: float = 1.0):
        self.name = name
        self.uses_rm = use_rm
        dim = features(_DummyCtx(), np.array([1.0])).shape[1]
        self.models = [_Ridge(dim, lam, sigma0=0.5) for _ in range(n_buckets)]
        self.t = [0] * n_buckets

    def _v(self, v_next):
        return v_next if self.uses_rm else np.zeros_like(v_next)

    def _lam(self, mod: _Ridge, eta: np.ndarray) -> np.ndarray:
        return np.maximum(np.exp(np.minimum(eta, 8.0) + 0.5 * mod.sigma**2) - 0.5, 0.0)

    def update(self, ctx, price, demand):
        self.models[ctx.bucket].update(features(ctx, np.array([price / ctx.static]))[0], float(np.log(demand + 0.5)))
        self.t[ctx.bucket] += 1

    def warm_start(self, logs) -> int:
        from .simulator import Context

        rows = logs[logs["price_test"] == 1]
        for r in rows.itertuples(index=False):
            ctx = Context(int(r.night), int(r.bucket), float(r.weekend), float(r.event), float(r.comp_price), float(r.static_price), float(r.pace))
            self.models[ctx.bucket].update(features(ctx, np.array([r.price / r.static_price]))[0], float(np.log(r.demand_unconstrained + 0.5)))
        return len(rows)


class EpsGreedy(LinearBandit):
    def __init__(self, name="eps_greedy", eps_cap=0.3, **kw):
        super().__init__(name, **kw)
        self.eps_cap = eps_cap

    def choose(self, ctx, cands, R, v_next, rng):
        t = self.t[ctx.bucket] + 1
        if rng.random() < min(self.eps_cap, t ** (-1 / 3)):
            return float(rng.choice(cands))
        mod = self.models[ctx.bucket]
        lam = self._lam(mod, features(ctx, cands / ctx.static) @ mod.theta)
        return float(cands[np.argmax(step_values(cands, lam, R, self._v(v_next)))])


class LinUCB(LinearBandit):
    def __init__(self, name="linucb", alpha=1.0, **kw):
        super().__init__(name, **kw)
        self.alpha = alpha

    def choose(self, ctx, cands, R, v_next, rng):
        mod = self.models[ctx.bucket]
        phi = features(ctx, cands / ctx.static)
        mean = phi @ mod.theta
        width = self.alpha * mod.sigma * np.sqrt(np.einsum("ij,jk,ik->i", phi, mod.Ainv, phi))
        v = self._v(v_next)
        hi = step_values(cands, self._lam(mod, mean + width), R, v)
        lo = step_values(cands, self._lam(mod, mean - width), R, v)
        return float(cands[np.argmax(np.maximum(hi, lo))])


class Thompson(LinearBandit):
    def __init__(self, name="thompson", scale=1.0, **kw):
        super().__init__(name, **kw)
        self.scale = scale

    def choose(self, ctx, cands, R, v_next, rng):
        mod = self.models[ctx.bucket]
        cov = (self.scale * mod.sigma) ** 2 * mod.Ainv
        L = np.linalg.cholesky(0.5 * (cov + cov.T) + 1e-9 * np.eye(len(cov)))
        theta = mod.theta + L @ rng.standard_normal(len(cov))
        lam = self._lam(mod, features(ctx, cands / ctx.static) @ theta)
        return float(cands[np.argmax(step_values(cands, lam, R, self._v(v_next)))])


class RandomPolicy(Policy):
    """Uniform over feasible candidates (used by tests and as an OPE logging policy)."""

    name = "random"

    def choose(self, ctx, cands, R, v_next, rng):
        return float(rng.choice(cands))


class _DummyCtx:
    def features(self):
        from .simulator import Context

        return Context(0, 0, 0.0, 0.0, 150.0, 150.0).features()
