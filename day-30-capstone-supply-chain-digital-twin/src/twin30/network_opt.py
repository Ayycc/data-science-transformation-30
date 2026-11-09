"""OR-Tools MILP for the network decision: open DC-East or not, and which DC serves each store.

The model is deterministic (expected annual demand from the posterior) and prices only what is
linear: DC fixed cost, outbound freight, DC handling, and in-transit plus safety stock holding
(proportional to transit days). Risk pooling, stockouts and supplier outages are left to the twin,
which re-evaluates every MILP answer under uncertainty.
"""

from __future__ import annotations

import numpy as np
from ortools.linear_solver import pywraplp

from .config import DCS, EXISTING_DCS, FINANCE, FREIGHT, N_DCS, N_STORES, UNIT_COST, geometry


def lane_costs(annual_units: np.ndarray) -> np.ndarray:
    """(D, S) annual variable cost of serving store s from DC d. annual_units: (S, C)."""
    geo = geometry()
    freight = geo.dist[:, :, None] * FREIGHT[None, None, :]  # (D, S, C) per unit
    handling = FINANCE.dc_handling_per_unit
    # pipeline + safety stock at the store grows with transit time: (transit + 1) days of cover
    days_cover = (geo.transit + 1)[:, :, None]
    holding = days_cover / 364.0 * UNIT_COST[None, None, :] * FINANCE.holding_rate_per_year
    return ((freight + handling) * annual_units[None, :, :] + holding * annual_units[None, :, :]).sum(axis=2)


def solve(annual_units: np.ndarray, allow_east: bool = True, force_east: bool = False) -> dict:
    """Minimise fixed + lane cost subject to single sourcing and DC daily throughput capacity."""
    cost = lane_costs(annual_units)
    load = annual_units.sum(axis=1) / 364.0  # units per day per store
    solver = pywraplp.Solver.CreateSolver("SCIP")
    y = [solver.BoolVar(f"open_{d}") for d in range(N_DCS)]
    x = [[solver.BoolVar(f"x_{d}_{s}") for s in range(N_STORES)] for d in range(N_DCS)]
    for d in EXISTING_DCS:
        solver.Add(y[d] == 1)
    if not allow_east:
        solver.Add(y[2] == 0)
    if force_east:
        solver.Add(y[2] == 1)
    for s in range(N_STORES):
        solver.Add(sum(x[d][s] for d in range(N_DCS)) == 1)
        for d in range(N_DCS):
            solver.Add(x[d][s] <= y[d])
    for d in range(N_DCS):
        solver.Add(sum(load[s] * x[d][s] for s in range(N_STORES)) <= FINANCE.dc_capacity_units_per_day[d] * y[d])
    fixed = FINANCE.dc_fixed_cost_per_year
    solver.Minimize(sum(fixed[d] * y[d] for d in range(N_DCS)) + sum(cost[d, s] * x[d][s] for d in range(N_DCS) for s in range(N_STORES)))
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"MILP not optimal: status {status}")
    assign = tuple(int(np.argmax([x[d][s].solution_value() for d in range(N_DCS)])) for s in range(N_STORES))
    return {
        "objective": solver.Objective().Value(),
        "open": [bool(round(v.solution_value())) for v in y],
        "assignment": assign,
        "assignment_names": [DCS[a].name for a in assign],
        "lane_cost": float(sum(cost[a, s] for s, a in enumerate(assign))),
        "status": "OPTIMAL",
        "wall_ms": solver.wall_time(),
    }


def evaluate(annual_units: np.ndarray, assignment: tuple[int, ...]) -> float:
    """MILP objective of a given assignment (used for the historical one)."""
    cost = lane_costs(annual_units)
    open_ = sorted(set(assignment) | set(EXISTING_DCS))
    return float(sum(FINANCE.dc_fixed_cost_per_year[d] for d in open_) + sum(cost[a, s] for s, a in enumerate(assignment)))
