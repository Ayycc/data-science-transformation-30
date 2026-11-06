"""Simulation study: run every policy on the same SYNTHETIC worlds over many seeds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.stats import poisson

from .policies import GRID, EpsGreedy, Guardrails, LinUCB, ModelPolicy, OraclePolicy, Policy, StaticPolicy, Thompson
from .rm import bid_price, night_value_tables, step_values
from .simulator import BUCKETS, MarketConfig, competitor_price, context_for, make_world, shock_posterior, true_demand_given_u, true_expected_demand

POLICY_ORDER = ("static", "greedy_naive", "greedy_iv", "eps_greedy", "linucb", "thompson", "linucb_no_rm", "oracle")
BANDITS = ("eps_greedy", "linucb", "thompson")


@dataclass
class StudyConfig:
    n_seeds: int = 20
    n_nights: int = 1095
    first_day: int = 365
    seed0: int = 1000
    policies: tuple = POLICY_ORDER
    linucb_alpha: float = 1.0
    ts_scale: float = 1.0
    guard: Guardrails = field(default_factory=Guardrails)
    warm_start: bool = True  # seed bandits with the randomised price-test cells of the logs


def make_policies(names, models: dict, cfg: MarketConfig, scfg: StudyConfig) -> dict[str, Policy]:
    makers: dict[str, Callable[[], Policy]] = {
        "static": StaticPolicy,
        "greedy_naive": lambda: ModelPolicy(models["naive_ols"], "greedy_naive"),
        "greedy_iv": lambda: ModelPolicy(models["iv_2sls"], "greedy_iv"),
        "eps_greedy": lambda: EpsGreedy(),
        "linucb": lambda: LinUCB(alpha=scfg.linucb_alpha),
        "thompson": lambda: Thompson(scale=scfg.ts_scale),
        "linucb_no_rm": lambda: LinUCB(name="linucb_no_rm", alpha=scfg.linucb_alpha, use_rm=False),
        "oracle": lambda: OraclePolicy(cfg),
    }
    pols = {n: makers[n]() for n in names}
    logs = models.get("logs")
    if scfg.warm_start and logs is not None:
        for p in pols.values():
            if hasattr(p, "warm_start"):
                p.warm_start(logs)
    return pols


def _rm_model_for(name: str) -> str:
    return {"greedy_naive": "naive_ols", "oracle": "truth"}.get(name, "iv_2sls")


def run_policy(policy: Policy, world, cfg: MarketConfig, guard: Guardrails, models: dict, uniforms: np.ndarray, seed: int) -> dict:
    """Simulate one policy over all nights of ``world``. ``uniforms[i, k]`` drive demand (common random numbers)."""
    K = len(BUCKETS)
    C = cfg.capacity
    rng = np.random.default_rng(seed)
    true_fn = lambda ctx, p: true_expected_demand(ctx, p, cfg)
    demand_fns = {"truth": true_fn, "iv_2sls": models["iv_2sls"].predict, "naive_ols": models["naive_ols"].predict}
    rm_key = _rm_model_for(policy.name)
    n = world.n
    prices = np.zeros((n, K))
    sold = np.zeros((n, K), dtype=int)
    demand = np.zeros((n, K), dtype=int)
    regret = np.zeros((n, K))
    bids = np.zeros((n, K))
    violations = 0
    prev: list[float | None] = [None] * K
    prev_ratio = 1.0
    for i in range(n):
        comp = competitor_price(world, i, prev_ratio)
        ctxs = [context_for(world, i, k, comp) for k in range(K)]
        V_ref = night_value_tables(ctxs, GRID, demand_fns["iv_2sls"], C)  # shared RM continuation value used to score regret
        V_pol = None
        if policy.uses_rm:
            V_pol = V_ref if rm_key == "iv_2sls" else night_value_tables(ctxs, GRID, demand_fns[rm_key], C)
        R = C
        history: list = []
        for k, ctx0 in enumerate(ctxs):
            w = shock_posterior(history, cfg) if history else None
            ctx = replace(ctx0, pace=float(np.log1p(sum(h[2] for h in history))), u_weights=w)
            cands = guard.candidates(ctx.static, prev[k])
            v_next = V_pol[k + 1] if V_pol is not None else np.zeros(C + 1)
            p = policy.choose(ctx, cands, R, v_next, rng)
            if not guard.is_ok(p, prev[k]):  # final safety layer; never expected to trigger
                violations += 1
                lo, hi = guard.bounds(prev[k])
                p = float(np.clip(p, lo, hi))
            lam = float(true_demand_given_u(ctx, np.array([p]), float(world.u[i]), cfg)[0])
            d = int(poisson.ppf(uniforms[i, k], lam)) if lam > 0 else 0
            s = min(d, R)
            # expected regret vs the best feasible price under the true (observable-context) demand curve
            g = step_values(cands, true_expected_demand(ctx, cands, cfg, w), R, V_ref[k + 1])
            j = int(np.argmin(np.abs(cands - p)))
            regret[i, k] = max(float(g.max() - g[j]), 0.0)
            bids[i, k] = bid_price(V_ref[k + 1], R) if R > 0 else np.nan
            prices[i, k], demand[i, k], sold[i, k] = p, d, s
            policy.update(ctx, p, d)
            history.append((ctx0, p, d))
            R -= s
            prev[k] = p
        prev_ratio = float(prices[i, 0] / ctxs[0].static)
    revenue_night = (prices * sold).sum(axis=1)
    rooms = sold.sum(axis=1)
    dlog = np.abs(np.diff(np.log(prices), axis=0))
    return dict(
        prices=prices,
        sold=sold,
        demand=demand,
        regret=regret,
        bids=bids,
        revenue=float(revenue_night.sum()),
        revpar=float(revenue_night.sum() / (C * n)),
        occupancy=float(rooms.sum() / (C * n)),
        adr=float(revenue_night.sum() / max(rooms.sum(), 1)),
        sellout_share=float(np.mean(rooms >= C)),
        cum_regret=np.cumsum(regret.sum(axis=1)),
        volatility=float(dlog.mean()),
        max_rel_change=float(np.max(np.abs(np.diff(prices, axis=0)) / prices[:-1])),
        min_price=float(prices.min()),
        max_price=float(prices.max()),
        guardrail_violations_posted=int(sum(not guard.is_ok(prices[i, k], prices[i - 1, k] if i else None) for i in range(n) for k in range(K))),
        guardrail_interventions=violations,
    )


def run_seed(seed: int, models: dict, cfg: MarketConfig, scfg: StudyConfig) -> dict:
    world = make_world(scfg.first_day, scfg.n_nights, seed, cfg)
    uniforms = np.random.default_rng(seed + 7).random((scfg.n_nights, len(BUCKETS)))
    pols = make_policies(scfg.policies, models, cfg, scfg)
    return {name: run_policy(pol, world, cfg, scfg.guard, models, uniforms, seed * 31 + j) for j, (name, pol) in enumerate(pols.items())}


def regret_exponent(cum: np.ndarray, start_frac: float = 0.25) -> float:
    """Slope of log cumulative regret on log t over the last (1 - start_frac) of the horizon; < 1 means sublinear."""
    t = np.arange(1, len(cum) + 1)
    s = int(len(cum) * start_frac)
    ok = cum[s:] > 0
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(t[s:][ok]), np.log(cum[s:][ok]), 1)[0])


def mean_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int = 10000) -> dict:
    """Mean with a t-interval and a percentile bootstrap interval (95%)."""
    from scipy.stats import t as tdist

    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    half = float(tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    boots = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    return dict(mean=m, t_ci=[m - half, m + half], boot_ci=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))], n=n)
