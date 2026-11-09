"""Scenario engine: plan-year context, Monte Carlo with common random numbers, risk metrics, optimiser."""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import network_opt
from . import promo as promo_mod
from .calibrate import Calibration
from .config import CATEGORIES, DAYS_PER_YEAR, DISCOUNT, HISTORICAL_ASSIGNMENT, PRICE, UNIT_COST, Decisions
from .demand import Posterior
from .twin import Scenario, StartState, simulate

STRESS_OUTAGE = (0, 14, 42)  # offshore supplier down for 28 days from plan day 14 (mid-November)
STRESS_EVERY = 4  # in the "mixed" regime every 4th replication carries the stress outage


@dataclass
class PlanContext:
    cal: Calibration
    posterior: Posterior
    start: StartState
    day0: int
    horizon: int
    calendars: dict  # strategy -> (promo (H, C), post (H, C))
    annual_units: np.ndarray  # (S, C) expected plan-year demand under the current calendar

    def scenario(self, dec: Decisions, regime: str, rep: int) -> Scenario:
        promo, post = self.calendars[dec.promo_strategy]
        forced = []
        if regime == "stress" or (regime == "mixed" and rep % STRESS_EVERY == 0):
            forced = [STRESS_OUTAGE]
        mode = "none" if regime == "calm" else "random"
        return Scenario(dec, self.day0, self.horizon, promo, post, mode, forced_outages=forced)


def posterior_mean_demand(post: Posterior, day0: int, n: int, promo: np.ndarray, postf: np.ndarray, k: int = 100) -> np.ndarray:
    idx = np.linspace(0, post.n - 1, min(k, post.n)).astype(int)
    return np.mean([post.mean(i, day0, n, promo, postf) for i in idx], axis=0)


def build_context(sales: pd.DataFrame, snapshot: dict, cal: Calibration, post: Posterior, day0: int, seed: int) -> PlanContext:
    H = DAYS_PER_YEAR
    n_weeks = [c.promo_weeks_per_year for c in CATEGORIES]
    # merchandisers' view of seasonality: last year's weekly sales index per category
    last = sales[(sales.day >= day0 - H) & (sales.day < day0)].copy()
    last["week"] = (last.day - (day0 - H)) // 7
    idx = last.pivot_table(index="week", columns="category", values="units_sold", aggfunc="sum")
    idx = idx[[c.name for c in CATEGORIES]].to_numpy()
    idx = idx / idx.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    cur_w = promo_mod.current_rule(idx, n_weeks, rng)
    calendars = {}
    zero = np.zeros((H, len(CATEGORIES)))
    calendars["none"] = (zero, zero)
    cur = promo_mod.weekly_to_daily(cur_w)
    calendars["current"] = (cur, promo_mod.post_promo(cur))
    # targeted: expected incremental gross margin of promoting each week, from the posterior
    base = posterior_mean_demand(post, day0, H, zero, zero).sum(axis=1)  # (H, C)
    wk = base.reshape(52, 7, -1).sum(axis=1)
    lift = np.exp(post.promo).mean(axis=0)
    dip = np.exp(post.post).mean(axis=0)
    m_reg = PRICE - UNIT_COST
    m_pro = PRICE * (1 - DISCOUNT) - UNIT_COST
    gain = wk * (lift * m_pro - m_reg)
    nxt = np.vstack([wk[1:], wk[-1:]])
    value = gain - nxt * (1 - dip) * m_reg
    tgt = promo_mod.weekly_to_daily(promo_mod.targeted_rule(value, n_weeks))
    calendars["targeted"] = (tgt, promo_mod.post_promo(tgt))
    annual = posterior_mean_demand(post, day0, H, *calendars["current"]).sum(axis=0)
    return PlanContext(cal, post, StartState.from_snapshot(snapshot), day0, H, calendars, annual)


# --- Monte Carlo ------------------------------------------------------------------------------
_CTX: PlanContext | None = None


def _one(args):
    dec, regime, rep, seed = args
    ctx = _CTX
    scen = dec if isinstance(dec, Scenario) else ctx.scenario(dec, regime, rep)
    return simulate(ctx.cal, ctx.posterior, scen, ctx.start, seed=seed)


def n_jobs() -> int:
    return max(1, int(os.environ.get("TWIN_JOBS", min(4, os.cpu_count() or 1))))


def run_many(ctx: PlanContext, jobs: list[tuple[Decisions, str, int, int]]) -> list[dict]:
    """jobs: (decisions, regime, rep, seed). Seeds are shared across decisions (common random numbers)."""
    global _CTX
    _CTX = ctx
    if n_jobs() == 1 or len(jobs) < 8:
        return [_one(j) for j in jobs]
    with mp.get_context("fork").Pool(n_jobs()) as pool:
        return pool.map(_one, jobs, chunksize=max(1, len(jobs) // (4 * n_jobs())))


def run_many_custom(ctx: PlanContext, scen: Scenario, reps: int, seed0: int) -> pd.DataFrame:
    """Run a fully specified Scenario (e.g. the validation replay) ``reps`` times."""
    return pd.DataFrame(run_many(ctx, [(scen, "custom", r, seed0 + r) for r in range(reps)]))


def run_scenario(ctx: PlanContext, dec: Decisions, regime: str, reps: int, seed0: int) -> pd.DataFrame:
    res = run_many(ctx, [(dec, regime, r, seed0 + r) for r in range(reps)])
    return pd.DataFrame(res)


# --- risk metrics -----------------------------------------------------------------------------
def cvar(x: np.ndarray, alpha: float = 0.05) -> float:
    """Expected value of the worst alpha share of outcomes (lower tail of profit)."""
    xs = np.sort(np.asarray(x, dtype=float))
    k = max(1, int(np.ceil(alpha * len(xs))))
    return float(xs[:k].mean())


def risk_summary(df: pd.DataFrame, base: pd.DataFrame | None = None) -> dict:
    p = df.operating_profit.to_numpy()
    out = {
        "reps": int(len(df)),
        "expected_profit": float(p.mean()),
        "profit_p5": float(np.percentile(p, 5)),
        "profit_p95": float(np.percentile(p, 95)),
        "cvar5_profit": cvar(p),
        "expected_revenue": float(df.revenue.mean()),
        "fill_rate_mean": float(df.fill_rate.mean()),
        "fill_rate_p5": float(np.percentile(df.fill_rate, 5)),
        "in_stock_rate_mean": float(df.in_stock_rate.mean()),
        "prob_fill_below_95": float((df.fill_rate < 0.95).mean()),
        "avg_dc_inventory_units": float(df.avg_dc_inventory_units.mean()),
        "avg_store_inventory_units": float(df.avg_store_inventory_units.mean()),
    }
    if base is not None:
        d = p - base.operating_profit.to_numpy()  # paired by seed (common random numbers)
        out["delta_profit_vs_baseline_mean"] = float(d.mean())
        out["delta_profit_vs_baseline_p5"] = float(np.percentile(d, 5))
        out["delta_profit_vs_baseline_p95"] = float(np.percentile(d, 95))
        out["prob_beats_baseline"] = float((d > 0).mean())
        out["delta_cvar5_vs_baseline"] = out["cvar5_profit"] - cvar(base.operating_profit.to_numpy())
    return out


# --- strategic scenarios ----------------------------------------------------------------------
def strategic_options(ctx: PlanContext) -> dict[str, dict]:
    """Baseline + three single-lever strategies; MILP chooses the store assignment for the new DC."""
    milp_status_quo = network_opt.solve(ctx.annual_units, allow_east=False)
    milp_east = network_opt.solve(ctx.annual_units, force_east=True)
    milp_free = network_opt.solve(ctx.annual_units, allow_east=True)
    base = Decisions()
    options = {
        "A_baseline": base,
        "B_new_dc_east": replace(base, open_dc_east=True, assignment=milp_east["assignment"]),
        "C_targeted_promos": replace(base, promo_strategy="targeted"),
        "D_dual_sourcing_40pct": replace(base, nearshore_share=0.40),
    }
    milp = {
        "status_quo_assignment_cost": network_opt.evaluate(ctx.annual_units, HISTORICAL_ASSIGNMENT),
        "best_without_east": milp_status_quo,
        "forced_east": milp_east,
        "free_choice": milp_free,
    }
    return {"options": options, "milp": milp}


def optimise(
    ctx: PlanContext,
    east_assignment: tuple[int, ...],
    base_screen: pd.DataFrame,
    screen_reps: int,
    final_reps: int,
    top_k: int,
    seed_screen: int,
    seed_final: int,
    regime: str = "mixed",
) -> dict:
    """Two-stage simulation optimisation over the decision grid.

    Stage 1 screens every combination on the same ``screen_reps`` seeds. Stage 2 re-runs the
    ``top_k`` feasible candidates (plus the baseline) on fresh seeds, so the winner's reported
    value is not inflated by having been selected on the same noise. Objective: maximise expected
    operating profit subject to CVaR5%(profit) >= the baseline's and mean fill rate >= 95%.
    """
    grid = {
        "open_dc_east": [False, True],
        "nearshore_share": [0.10, 0.25, 0.40],
        "store_z": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
        "promo_strategy": ["current", "targeted"],
    }
    combos = []
    for east, share, z, promo in itertools.product(*grid.values()):
        combos.append(
            Decisions(
                open_dc_east=east,
                assignment=east_assignment if east else HISTORICAL_ASSIGNMENT,
                nearshore_share=share,
                store_z=z,
                promo_strategy=promo,
            )
        )
    jobs = [(d, regime, r, seed_screen + r) for d in combos for r in range(screen_reps)]
    res = run_many(ctx, jobs)
    base_cvar = cvar(base_screen.operating_profit.to_numpy())
    rows = []
    for i, d in enumerate(combos):
        df = pd.DataFrame(res[i * screen_reps : (i + 1) * screen_reps])
        s = risk_summary(df, base_screen)
        rows.append(
            {
                "open_dc_east": d.open_dc_east,
                "nearshore_share": d.nearshore_share,
                "store_z": d.store_z,
                "promo_strategy": d.promo_strategy,
                **{k: s[k] for k in ("expected_profit", "cvar5_profit", "fill_rate_mean", "delta_profit_vs_baseline_mean")},
                "feasible": bool(s["cvar5_profit"] >= base_cvar and s["fill_rate_mean"] >= 0.95),
            }
        )
    screen = pd.DataFrame(rows)
    cand = screen[screen.feasible].sort_values("expected_profit", ascending=False).head(top_k)
    finals = []
    base_dec = Decisions()
    base_final = run_scenario(ctx, base_dec, regime, final_reps, seed_final)
    base_final_cvar = cvar(base_final.operating_profit.to_numpy())
    for idx in cand.index:
        d = combos[idx]
        df = run_scenario(ctx, d, regime, final_reps, seed_final)
        s = risk_summary(df, base_final)
        s["feasible"] = bool(s["cvar5_profit"] >= base_final_cvar and s["fill_rate_mean"] >= 0.95)
        finals.append((d, s))
    feasible = [f for f in finals if f[1]["feasible"]]
    best = max(feasible, key=lambda f: f[1]["expected_profit"]) if feasible else (base_dec, risk_summary(base_final, base_final))
    return {
        "grid": grid,
        "n_combinations": len(combos),
        "screen_reps": screen_reps,
        "final_reps": final_reps,
        "regime": regime,
        "screen": screen,
        "finalists": [
            {
                "open_dc_east": d.open_dc_east,
                "nearshore_share": d.nearshore_share,
                "store_z": d.store_z,
                "promo_strategy": d.promo_strategy,
                **s,
            }
            for d, s in finals
        ],
        "best_decisions": best[0],
        "best_summary": best[1],
        "baseline_final_summary": risk_summary(base_final),
    }
