"""SimPy discrete-event simulation of weekly review, stochastic supplier lead times and lost sales.

Time unit is one week. Each week at t = k the planner reviews inventory positions and places orders (subject to
budget, warehouse capacity and MOQ); every order spawns a SimPy delivery process that waits its sampled lead time.
Demand for week k is served at t = k + 0.5 from on-hand stock (unmet demand is lost), and holding cost is charged
on the stock left at t = k + 1. Seasonal items get a single pre-season buy and are salvaged after the season.
All policies see the same demand and the same lead-time draws (common random numbers). The first ``warmup`` weeks
replay the end of the history so every policy reaches its own steady state; only the test year is scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import simpy

from .milp import Instance, greedy, solve_ortools


@dataclass
class SimInputs:
    demand: np.ndarray  # items x weeks (test year actuals)
    lead_times: np.ndarray  # items x weeks, weeks
    s: np.ndarray
    S: np.ndarray
    pack: np.ndarray
    moq: np.ndarray
    cost: np.ndarray
    price: np.ndarray
    vol: np.ndarray
    wh: np.ndarray
    h_week: np.ndarray
    shortage: np.ndarray
    K: float
    budget: float
    capacity: np.ndarray  # m3 per warehouse
    init_on_hand: np.ndarray
    seasonal: np.ndarray  # bool
    season_qty: np.ndarray  # pre-season buy per item (0 for non-seasonal)
    season_order_week: np.ndarray  # week index (-1 = none)
    season_end_week: int
    salvage: np.ndarray  # $ recovered per leftover seasonal unit
    w_s: np.ndarray
    w_S: np.ndarray
    warmup: int = 0  # leading columns of ``demand`` replayed from history to reach steady state; not scored


@dataclass
class SimResult:
    fill_rate: float
    served: np.ndarray
    demand: np.ndarray
    holding_cost: float
    shortage_cost: float
    ordering_cost: float
    overage_cost: float
    purchase_spend: float
    avg_inventory_value: float
    n_order_lines: int
    weekly_inventory_value: np.ndarray
    weekly_spend: np.ndarray
    weekly_capacity_util: np.ndarray  # max over warehouses of (on hand + on order) m3 / capacity after ordering
    milp_stats: list = field(default_factory=list)
    on_hand_trace: np.ndarray | None = None
    avg_on_hand: np.ndarray | None = None  # per item, scored weeks
    first_instance: tuple | None = None  # (Instance, IP, item positions, Solution) at the first scored week

    @property
    def total_cost(self) -> float:
        return self.holding_cost + self.shortage_cost + self.ordering_cost + self.overage_cost


def simulate(
    inp: SimInputs,
    allocator: str = "milp",
    time_limit_s: float = 10.0,
    rel_gap: float = 1e-4,
    trace_items: np.ndarray | None = None,
) -> SimResult:
    """Run one year. ``allocator`` is "milp" (OR-Tools) or "greedy" (planner order-of-list truncation)."""
    n, T = inp.demand.shape
    env = simpy.Environment()
    on_hand = inp.init_on_hand.astype(float).copy()
    on_order = np.zeros(n)
    served = np.zeros(n)
    oh_sum = np.zeros(n)
    acc = {"hold": 0.0, "short": 0.0, "order": 0.0, "over": 0.0, "spend": 0.0, "lines": 0}
    inv_val = np.zeros(T)
    spend_w = np.zeros(T)
    cap_util = np.zeros(T)
    stats: list[dict] = []
    first_instance: list = []
    trace = np.zeros((len(trace_items), T)) if trace_items is not None else None
    regular = ~inp.seasonal
    W = inp.warmup
    n_wh = len(inp.capacity)

    def delivery(i: int, q: float, lt: float):
        yield env.timeout(lt)
        on_hand[i] += q
        on_order[i] -= q

    def place(i: int, q: float, k: int, regular_buy: bool = True) -> None:
        on_order[i] += q
        if regular_buy:  # the weekly budget governs regular replenishment; seasonal buys are approved separately
            spend_w[k] += q * inp.cost[i]
        if k >= W:
            acc["order"] += inp.K
            acc["spend"] += q * inp.cost[i]
            acc["lines"] += 1
        env.process(delivery(i, q, float(inp.lead_times[i, k])))

    def week_cycle():
        for k in range(T):
            # 1) review
            for i in np.where(inp.season_order_week == k)[0]:
                if inp.season_qty[i] > 0:
                    place(int(i), float(inp.season_qty[i]), k, regular_buy=False)
            ip = on_hand + on_order
            trig = np.where(regular & (inp.S > 0) & (ip <= inp.s))[0]
            if len(trig):
                vol_now = np.bincount(inp.wh, weights=(on_hand + on_order) * inp.vol, minlength=n_wh)
                inst = Instance(
                    gap_s=inp.s[trig] + 1 - ip[trig],
                    gap_S=inp.S[trig] - ip[trig],
                    pack=inp.pack[trig],
                    moq=inp.moq[trig],
                    cost=inp.cost[trig],
                    vol=inp.vol[trig],
                    wh=inp.wh[trig],
                    w_s=inp.w_s[trig],
                    w_S=inp.w_S[trig],
                    h=inp.h_week[trig],
                    K=inp.K,
                    budget=inp.budget,
                    cap_left=inp.capacity - vol_now,
                )
                if allocator == "milp":
                    sol = solve_ortools(inst, time_limit_s=time_limit_s, rel_gap=rel_gap)
                    q = sol.q
                    if k == W:
                        first_instance.append((inst, ip[trig].copy(), trig.copy(), sol))
                    stats.append(
                        {
                            "week": k,
                            "n_vars": 4 * inst.n,
                            "triggered": inst.n,
                            "ordered": int((q > 0).sum()),
                            "status": sol.status,
                            "gap": sol.gap,
                            "seconds": sol.seconds,
                            "spend": float((q * inst.cost).sum()),
                            "budget": inst.budget,
                        }
                    )
                else:
                    q = greedy(inst)
                for j, i in enumerate(trig):
                    if q[j] > 0:
                        place(int(i), float(q[j]), k)
            vol_after = np.bincount(inp.wh, weights=(on_hand + on_order) * inp.vol, minlength=n_wh)
            cap_util[k] = float(np.max(vol_after / inp.capacity))
            yield env.timeout(0.5)
            # 2) demand (lost sales)
            d = inp.demand[:, k]
            sv = np.minimum(on_hand, d)
            on_hand[:] = on_hand - sv
            scored = k >= W
            if scored:
                served[:] = served + sv
                acc["short"] += float(np.sum((d - sv) * inp.shortage))
            if k == inp.season_end_week:
                left = on_hand * inp.seasonal
                acc["over"] += float(np.sum(left * (inp.cost - inp.salvage)))
                on_hand[:] = on_hand - left
            yield env.timeout(0.5)
            # 3) end-of-week holding
            if scored:
                acc["hold"] += float(np.sum(on_hand * inp.h_week))
                oh_sum[:] = oh_sum + on_hand
            inv_val[k] = float(np.sum(on_hand * inp.cost))
            if trace is not None:
                trace[:, k] = on_hand[trace_items]

    env.process(week_cycle())
    env.run(until=T)
    tot_d = inp.demand[:, W:].sum(axis=1).astype(float)
    return SimResult(
        fill_rate=float(served.sum() / max(tot_d.sum(), 1)),
        served=served,
        demand=tot_d,
        holding_cost=acc["hold"],
        shortage_cost=acc["short"],
        ordering_cost=acc["order"],
        overage_cost=acc["over"],
        purchase_spend=acc["spend"],
        avg_inventory_value=float(inv_val[W:].mean()),
        n_order_lines=int(acc["lines"]),
        weekly_inventory_value=inv_val[W:],
        weekly_spend=spend_w[W:],
        weekly_capacity_util=cap_util[W:],
        milp_stats=[r for r in stats if r["week"] >= W],
        on_hand_trace=trace[:, W:] if trace is not None else None,
        first_instance=first_instance[0] if first_instance else None,
        avg_on_hand=oh_sum / max(T - W, 1),
    )
