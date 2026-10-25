from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bikeshare import routing
from bikeshare import simulate as SIM
from bikeshare.config import FleetConfig


def _cand(n=30, seed=0):
    rng = np.random.default_rng(seed)
    move = rng.integers(3, 15, n) * rng.choice([-1, 1], n)
    return pd.DataFrame(
        {"station": np.arange(n), "lat": 38.9 + rng.uniform(-0.03, 0.03, n), "lon": -77.03 + rng.uniform(-0.04, 0.04, n), "move": move}
    )


@pytest.mark.parametrize("shift_min", [60, 240])
def test_routes_respect_capacity_and_shift(shift_min):
    cfg = FleetConfig(n_trucks=3, truck_capacity=12)
    cand = _cand()
    plan = routing.solve((38.9, -77.03), cand, cfg, shift_min * 60, solution_limit=100)
    chk = routing.check_plan(plan, cfg, shift_min * 60)
    assert chk["feasible"]
    assert chk["max_load"] <= 12 and chk["max_duration_s"] <= shift_min * 60
    assert plan.planned_bikes > 0 and len(plan.routes) <= 3
    for stops in plan.routes:
        assert sum(s.move for s in stops) == 0  # trucks return empty
        assert all(s.eta_s <= shift_min * 60 for s in stops)


def test_longer_shift_serves_more():
    cfg = FleetConfig(n_trucks=2, truck_capacity=12)
    cand = _cand(seed=3)
    short = routing.solve((38.9, -77.03), cand, cfg, 45 * 60, solution_limit=100)
    long = routing.solve((38.9, -77.03), cand, cfg, 300 * 60, solution_limit=100)
    assert long.planned_bikes > short.planned_bikes


def test_target_levels_band():
    cap = np.array([20, 20, 20])
    bikes = np.array([1, 19, 10])
    tgt, move = SIM.target_levels(bikes, cap, drain=np.array([6.0, 0, 30]), fill=np.array([0.0, 8, 0]), buffer=2)
    assert tgt[0] == 8 and move[0] == 7  # needs 2 + 6 bikes at least
    assert tgt[1] == 10 and move[1] == -9  # needs 2 + 8 docks free
    assert tgt[2] == 20  # infeasible band -> midpoint, clipped to capacity
    d, f = SIM.path_band(np.array([[-1.0, -2, 4, -1]]))
    assert d[0] == 3 and f[0] == 1


def test_simulator_counts_stockouts_and_rebalancing():
    st = pd.DataFrame({"station_id": ["A", "B"], "lat": [38.90, 38.91], "lon": [-77.03, -77.03], "capacity": [4, 4]})
    start = pd.Timestamp("2024-09-16")
    t = [start + pd.Timedelta(hours=7, minutes=m) for m in range(0, 50, 10)]
    # five rides A -> B in the morning: A starts with 2 bikes, B with 2 free docks
    trips = pd.DataFrame({"start_time": t, "end_time": [x + pd.Timedelta(minutes=5) for x in t], "start_id": "A", "end_id": "B"})
    cfg = FleetConfig(n_trucks=1, truck_capacity=10, shifts=(("overnight", 0, 300, 14),))
    none = SIM.simulate(st, trips, SIM.Policy(), start, start + pd.Timedelta(days=1), cfg)
    s = none.summary()
    assert s["lost_departures"] == 3 and s["diverted_arrivals"] == 0
    assert none.empty_min[0] > 0 and none.full_min[1] > 0

    class Move(SIM.Policy):
        def plan(self, t0, shift, bikes):
            p = routing.RoutePlan()
            p.routes = [[routing.Stop(station=1, move=-2, eta_s=600), routing.Stop(station=0, move=2, eta_s=1200)]]
            p.durations_s = [1800]
            return p

    reb = SIM.simulate(st, trips, Move(), start, start + pd.Timedelta(days=1), cfg)
    assert reb.bikes_moved == 4
    assert reb.summary()["lost_departures"] == 1
    assert reb.full_min[1] < none.full_min[1]  # B no longer overflows in the morning
