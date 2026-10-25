"""Capacitated pickup-and-delivery rebalancing routes with OR-Tools.

Each candidate station has a signed move: ``move > 0`` delivers bikes (truck load falls), ``move < 0``
picks bikes up (load rises). Trucks leave the depot empty and must return empty, so a route is a
pure redistribution. Load must stay within [0, truck capacity] at every stop and the route
(driving + handling) must fit inside the shift. Stations may be skipped at a penalty proportional to
the bikes they needed, so the solver serves the most valuable imbalance first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .config import FleetConfig
from .geo import haversine_matrix


@dataclass
class Stop:
    station: int  # index into the station table
    move: int  # planned signed move (+ deliver, - pick up)
    eta_s: int  # seconds after shift start
    target: int | None = None  # level-to-target instruction (fixed-route policy only)


@dataclass
class RoutePlan:
    routes: list[list[Stop]] = field(default_factory=list)
    durations_s: list[int] = field(default_factory=list)
    distance_m: float = 0.0
    planned_bikes: int = 0
    candidates: int = 0
    dropped: int = 0
    max_load: int = 0


def travel_seconds(lat: np.ndarray, lon: np.ndarray, cfg: FleetConfig) -> tuple[np.ndarray, np.ndarray]:
    """Road-distance proxy (straight line x circuity) and drive time in seconds."""
    d = haversine_matrix(lat, lon, lat, lon) * cfg.circuity
    return d, np.round(d / (cfg.speed_kmh / 3.6)).astype(np.int64)


def solve(depot: tuple[float, float], cand: pd.DataFrame, cfg: FleetConfig, shift_s: int, solution_limit: int = 300) -> RoutePlan:
    """``cand`` has columns station, lat, lon, move. Returns routes that respect capacity and shift length."""
    cand = cand[cand["move"].abs() >= 1].reset_index(drop=True)
    plan = RoutePlan(candidates=len(cand))
    if cand.empty:
        return plan
    move = np.clip(cand["move"].to_numpy(int), -cfg.truck_capacity, cfg.truck_capacity)
    lat = np.r_[depot[0], cand["lat"].to_numpy(float)]
    lon = np.r_[depot[1], cand["lon"].to_numpy(float)]
    dist_m, drive = travel_seconds(lat, lon, cfg)
    service = np.r_[0, cfg.service_s + cfg.per_bike_s * np.abs(move)].astype(np.int64)
    load_delta = np.r_[0, -move].astype(np.int64)  # pickup increases load

    n = len(lat)
    mgr = pywrapcp.RoutingIndexManager(n, cfg.n_trucks, 0)
    routing = pywrapcp.RoutingModel(mgr)

    def time_cb(i: int, j: int) -> int:
        a, b = mgr.IndexToNode(i), mgr.IndexToNode(j)
        return int(drive[a, b] + service[a])

    def dist_cb(i: int, j: int) -> int:
        return int(drive[mgr.IndexToNode(i), mgr.IndexToNode(j)])

    def load_cb(i: int) -> int:
        return int(load_delta[mgr.IndexToNode(i)])

    t_idx = routing.RegisterTransitCallback(time_cb)
    d_idx = routing.RegisterTransitCallback(dist_cb)
    l_idx = routing.RegisterUnaryTransitCallback(load_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(d_idx)
    routing.AddDimension(t_idx, 0, shift_s, True, "Time")
    routing.AddDimensionWithVehicleCapacity(l_idx, 0, [cfg.truck_capacity] * cfg.n_trucks, True, "Load")
    load_dim = routing.GetDimensionOrDie("Load")
    time_dim = routing.GetDimensionOrDie("Time")
    for v in range(cfg.n_trucks):
        load_dim.CumulVar(routing.End(v)).SetRange(0, 0)  # return empty: pure redistribution
    for node in range(1, n):
        routing.AddDisjunction([mgr.NodeToIndex(node)], int(cfg.penalty_per_bike_s * abs(move[node - 1])))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.solution_limit = solution_limit  # deterministic stopping rule
    params.time_limit.FromSeconds(int(max(1, cfg.solver_seconds * 10)))  # safety cap only
    sol = routing.SolveWithParameters(params)
    if sol is None:
        plan.dropped = len(cand)
        return plan

    served = 0
    for v in range(cfg.n_trucks):
        idx = routing.Start(v)
        stops: list[Stop] = []
        prev = 0
        dist = 0.0
        while not routing.IsEnd(idx):
            idx = sol.Value(routing.NextVar(idx))
            node = mgr.IndexToNode(idx)
            dist += dist_m[prev, node]
            prev = node
            if node != 0:
                eta = sol.Value(time_dim.CumulVar(idx))
                stops.append(Stop(station=int(cand.at[node - 1, "station"]), move=int(move[node - 1]), eta_s=int(eta)))
                plan.max_load = max(plan.max_load, int(sol.Value(load_dim.CumulVar(idx)) + load_delta[node]))
        if stops:
            plan.routes.append(stops)
            plan.durations_s.append(int(sol.Value(time_dim.CumulVar(routing.End(v)))))
            plan.distance_m += dist
            served += len(stops)
            plan.planned_bikes += int(sum(abs(s.move) for s in stops))
    plan.dropped = len(cand) - served
    return plan


def check_plan(plan: RoutePlan, cfg: FleetConfig, shift_s: int) -> dict:
    """Independent re-check of every route: running load within [0, Q], empty at return, within shift."""
    worst_load, worst_dur, ok = 0, 0, True
    for stops, dur in zip(plan.routes, plan.durations_s, strict=True):
        load = 0
        for s in stops:
            load -= s.move
            worst_load = max(worst_load, load)
            ok &= 0 <= load <= cfg.truck_capacity
        ok &= load == 0
        ok &= dur <= shift_s
        worst_dur = max(worst_dur, dur)
    ok &= len(plan.routes) <= cfg.n_trucks
    return {"feasible": bool(ok), "max_load": int(worst_load), "max_duration_s": int(worst_dur)}
