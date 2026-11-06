"""Stretch goal: off-policy evaluation of pricing rules from the randomised price-test logs.

Logging policy: on a test cell the multiplier was uniform over TEST_MULTS (propensity 1/5).
Reward of a cell: unconstrained room revenue price x demand (capacity ignored, so cells are
independent). Target policies map a context to one of the test multipliers. We compare IPS,
self-normalised IPS, a direct method (ridge reward model fitted on the same cells) and doubly
robust estimates with the simulator's true expected value on the same cells.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .estimation import TEST_MULTS
from .simulator import Context, MarketConfig, true_demand_given_u


def _ctx(r) -> Context:
    return Context(int(r.night), int(r.bucket), float(r.weekend), float(r.event), float(r.comp_price), float(r.static_price), float(r.pace))


def _phi(ctx: Context, m: float) -> np.ndarray:
    x = ctx.features()
    b = np.zeros(3)
    b[ctx.bucket] = 1.0
    return np.concatenate([np.kron(b, x), np.kron(b, x) * m, np.kron(b, x) * m * m])


def fit_reward_model(cells: pd.DataFrame, lam: float = 1.0) -> Callable[[Context, float], float]:
    """Direct-method model: ridge on log(demand + 0.5), revenue = price * exp(.) with smearing."""
    ctxs = [_ctx(r) for r in cells.itertuples(index=False)]
    X = np.array([_phi(c, r.test_multiplier) for c, r in zip(ctxs, cells.itertuples(index=False))])
    y = np.log(cells["demand_unconstrained"].to_numpy() + 0.5)
    theta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    smear = float(np.mean(np.exp(y - X @ theta)))
    return lambda ctx, m: float(ctx.static * m * max(np.exp(_phi(ctx, m) @ theta) * smear - 0.5, 0.0))


def evaluate(logs: pd.DataFrame, targets: dict[str, Callable[[Context], float]], cfg: MarketConfig, hidden_u: np.ndarray, seed: int = 0) -> dict:
    cells = logs[logs["price_test"] == 1].reset_index(drop=True)
    ctxs = [_ctx(r) for r in cells.itertuples(index=False)]
    m_log = cells["test_multiplier"].to_numpy()
    r_log = (cells["price"] * cells["demand_unconstrained"]).to_numpy()
    prop = 1.0 / len(TEST_MULTS)
    q = fit_reward_model(cells)
    u = hidden_u[cells["night"].to_numpy()]
    rng = np.random.default_rng(seed)
    out = {}
    for name, pi in targets.items():
        a = np.array([pi(c) for c in ctxs])
        match = np.isclose(a, m_log)
        w = match / prop
        dm_pi = np.array([q(c, ai) for c, ai in zip(ctxs, a)])
        dm_log = np.array([q(c, mi) for c, mi in zip(ctxs, m_log)])
        ips = w * r_log
        dr = dm_pi + w * (r_log - dm_log)
        truth = np.array([c.static * ai * true_demand_given_u(c, np.array([c.static * ai]), ui, cfg)[0] for c, ai, ui in zip(ctxs, a, u)])
        boot = rng.integers(0, len(cells), size=(2000, len(cells)))
        res = dict(
            truth=float(truth.mean()),
            ips=float(ips.mean()),
            snips=float((w * r_log).sum() / max(w.sum(), 1e-9)),
            dm=float(dm_pi.mean()),
            dr=float(dr.mean()),
            ips_ci=[float(np.quantile(ips[boot].mean(1), 0.025)), float(np.quantile(ips[boot].mean(1), 0.975))],
            dr_ci=[float(np.quantile(dr[boot].mean(1), 0.025)), float(np.quantile(dr[boot].mean(1), 0.975))],
            matched_cells=int(match.sum()),
        )
        for k in ("ips", "snips", "dm", "dr"):
            res[f"{k}_rel_error"] = (res[k] - res["truth"]) / res["truth"]
        out[name] = res
    out["_n_cells"] = len(cells)
    return out
