"""Weekly multi-warehouse replenishment MILP (OR-Tools SCIP; PuLP/CBC cross-check).

Given the items whose inventory position is at or below s, choose integer pack quantities subject to
MOQ (semi-continuous: 0 or >= MOQ), a weekly purchase budget and per-warehouse volume capacity. The objective
trades ordering cost against units left below the reorder point s (heavily weighted) and below the
order-up-to level S (lightly weighted), each priced at the item's shortage cost, plus one week of holding cost
on what is bought.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Instance:
    gap_s: np.ndarray  # (s + 1 - IP) units needed to get back above the reorder point
    gap_S: np.ndarray  # (S - IP) units needed to reach the order-up-to level
    pack: np.ndarray
    moq: np.ndarray
    cost: np.ndarray  # unit purchase cost
    vol: np.ndarray  # m3 per unit
    wh: np.ndarray  # warehouse index
    w_s: np.ndarray  # $ weight per unit below s
    w_S: np.ndarray  # $ weight per unit below S
    h: np.ndarray  # $ holding per unit-week
    K: float
    budget: float
    cap_left: np.ndarray  # m3 left per warehouse

    @property
    def n(self) -> int:
        return len(self.gap_S)

    def upper_packs(self) -> np.ndarray:
        """Never buy more than needed to reach S, except for pack/MOQ rounding."""
        need = np.ceil(np.maximum(self.gap_S, 0) / self.pack)
        return np.maximum(need, np.ceil(self.moq / self.pack)).astype(int)


@dataclass
class Solution:
    q: np.ndarray
    status: str
    objective: float
    bound: float
    gap: float
    seconds: float


def _gap(obj: float, bound: float) -> float:
    return abs(obj - bound) / max(abs(obj), 1e-9) if np.isfinite(bound) else float("nan")


def solve_ortools(inst: Instance, time_limit_s: float = 20.0, rel_gap: float = 0.0, backend: str = "SCIP") -> Solution:
    from ortools.linear_solver import pywraplp

    t0 = time.perf_counter()
    if inst.n == 0:
        return Solution(np.zeros(0, int), "OPTIMAL", 0.0, 0.0, 0.0, 0.0)
    solver = pywraplp.Solver.CreateSolver(backend)
    solver.SetTimeLimit(int(time_limit_s * 1000))
    ub = inst.upper_packs()
    n = [solver.IntVar(0, int(ub[i]), f"n{i}") for i in range(inst.n)]
    y = [solver.BoolVar(f"y{i}") for i in range(inst.n)]
    v = [solver.NumVar(0, solver.infinity(), f"v{i}") for i in range(inst.n)]
    u = [solver.NumVar(0, solver.infinity(), f"u{i}") for i in range(inst.n)]
    for i in range(inst.n):
        q = inst.pack[i] * n[i]
        solver.Add(q >= inst.moq[i] * y[i])
        solver.Add(n[i] <= int(ub[i]) * y[i])
        solver.Add(v[i] >= inst.gap_s[i] - q)
        solver.Add(u[i] >= inst.gap_S[i] - q)
    solver.Add(sum(inst.cost[i] * inst.pack[i] * n[i] for i in range(inst.n)) <= inst.budget)
    for w in np.unique(inst.wh):
        idx = np.where(inst.wh == w)[0]
        solver.Add(sum(inst.vol[i] * inst.pack[i] * n[i] for i in idx) <= max(float(inst.cap_left[w]), 0.0))
    obj = solver.Objective()
    for i in range(inst.n):
        obj.SetCoefficient(y[i], inst.K)
        obj.SetCoefficient(v[i], float(inst.w_s[i]))
        obj.SetCoefficient(u[i], float(inst.w_S[i]))
        obj.SetCoefficient(n[i], float(inst.h[i] * inst.pack[i]))
    obj.SetMinimization()
    params = pywraplp.MPSolverParameters()
    params.SetDoubleParam(pywraplp.MPSolverParameters.RELATIVE_MIP_GAP, rel_gap)
    status = solver.Solve(params)
    names = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE"}
    st = names.get(status, f"STATUS_{status}")
    if st not in ("OPTIMAL", "FEASIBLE"):
        return Solution(np.zeros(inst.n, int), st, float("nan"), float("nan"), float("nan"), time.perf_counter() - t0)
    q = np.array([round(n[i].solution_value()) * inst.pack[i] for i in range(inst.n)], dtype=np.int64)
    val, bound = obj.Value(), obj.BestBound()
    gap = 0.0 if st == "OPTIMAL" and _gap(val, bound) < 1e-6 else _gap(val, bound)
    return Solution(q, st, val, bound, gap, time.perf_counter() - t0)


def solve_pulp(inst: Instance, time_limit_s: float = 20.0) -> Solution:
    import pulp

    t0 = time.perf_counter()
    prob = pulp.LpProblem("replenishment", pulp.LpMinimize)
    ub = inst.upper_packs()
    n = [pulp.LpVariable(f"n{i}", 0, int(ub[i]), cat="Integer") for i in range(inst.n)]
    y = [pulp.LpVariable(f"y{i}", cat="Binary") for i in range(inst.n)]
    v = [pulp.LpVariable(f"v{i}", 0) for i in range(inst.n)]
    u = [pulp.LpVariable(f"u{i}", 0) for i in range(inst.n)]
    prob += pulp.lpSum(
        inst.K * y[i] + float(inst.w_s[i]) * v[i] + float(inst.w_S[i]) * u[i] + float(inst.h[i] * inst.pack[i]) * n[i]
        for i in range(inst.n)
    )
    for i in range(inst.n):
        q = int(inst.pack[i]) * n[i]
        prob += q >= int(inst.moq[i]) * y[i]
        prob += n[i] <= int(ub[i]) * y[i]
        prob += v[i] >= float(inst.gap_s[i]) - q
        prob += u[i] >= float(inst.gap_S[i]) - q
    prob += pulp.lpSum(float(inst.cost[i] * inst.pack[i]) * n[i] for i in range(inst.n)) <= inst.budget
    for w in np.unique(inst.wh):
        idx = np.where(inst.wh == w)[0]
        prob += pulp.lpSum(float(inst.vol[i] * inst.pack[i]) * n[i] for i in idx) <= max(float(inst.cap_left[w]), 0.0)
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s, gapRel=0.0))
    # PuLP reports "Optimal" for CBC runs stopped by the time limit; sol_status separates proven optimum (1)
    # from an integer-feasible incumbent (2).
    st = {1: "OPTIMAL", 2: "FEASIBLE_TIME_LIMIT"}.get(prob.sol_status, pulp.LpStatus[prob.status].upper())
    q = np.array([round(n[i].value() or 0) * inst.pack[i] for i in range(inst.n)], dtype=np.int64)
    return Solution(q, st, float(pulp.value(prob.objective) or 0.0), float("nan"), float("nan"), time.perf_counter() - t0)


def greedy(inst: Instance) -> np.ndarray:
    """Planner allocation: walk orders in list order, keep each one that still fits budget and capacity."""
    want = np.where(inst.gap_S > 0, np.maximum(np.ceil(inst.gap_S / inst.pack) * inst.pack, inst.moq), 0).astype(np.int64)
    q = np.zeros(inst.n, np.int64)
    budget = inst.budget
    cap = inst.cap_left.astype(float).copy()
    for i in range(inst.n):
        spend, vol = want[i] * inst.cost[i], want[i] * inst.vol[i]
        if want[i] > 0 and spend <= budget and vol <= cap[inst.wh[i]]:
            q[i] = want[i]
            budget -= spend
            cap[inst.wh[i]] -= vol
    return q


def objective_value(inst: Instance, q: np.ndarray) -> float:
    v = np.maximum(inst.gap_s - q, 0)
    u = np.maximum(inst.gap_S - q, 0)
    return float(np.sum(inst.K * (q > 0) + inst.w_s * v + inst.w_S * u + inst.h * q))
