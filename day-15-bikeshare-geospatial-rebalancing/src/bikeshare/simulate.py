"""Event-driven dock simulation of the test fortnight under different rebalancing policies.

Real (or SYNTHETIC) trips are replayed in time order against dock inventories:
- a departure from an empty station is a lost trip (its arrival is cancelled too);
- an arrival at a full station is diverted to the nearest station with a free dock.
Trucks execute their stops at the solver's ETAs; every move is clipped to what is physically possible
(bikes on the truck, bikes at the station, free docks, truck capacity), so planned and executed moves
can differ. Empty-dock and full-dock minutes are integrated per station over the whole fortnight.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import routing
from .config import FleetConfig

ShiftSpec = tuple[str, int, int, int]


def path_band(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(drain, fill) of an hourly net-flow path: deepest dip and highest rise of its running sum."""
    cum = np.cumsum(path, axis=1)
    return -np.minimum(0.0, cum.min(axis=1)), np.maximum(0.0, cum.max(axis=1))


def target_levels(bikes: np.ndarray, cap: np.ndarray, drain: np.ndarray, fill: np.ndarray, buffer: int) -> tuple[np.ndarray, np.ndarray]:
    """Inventory band [buffer + drain, cap - buffer - fill] that survives the horizon.

    Returns (target, move): the nearest level inside the band (or its midpoint when the band is empty).
    """
    lo = buffer + drain
    hi = cap - buffer - fill
    feasible = lo <= hi
    tgt = np.where(feasible, np.clip(bikes, lo, hi), (lo + hi) / 2.0)
    tgt = np.clip(np.round(tgt), 0, cap).astype(int)
    return tgt, tgt - bikes


def candidates(stations: pd.DataFrame, move: np.ndarray, cfg: FleetConfig, target: np.ndarray | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"station": np.arange(len(move)), "lat": stations["lat"].to_numpy(), "lon": stations["lon"].to_numpy(), "move": move})
    if target is not None:
        df["target"] = target
    df = df[df["move"].abs() >= cfg.min_move]
    df = df.reindex(df["move"].abs().sort_values(ascending=False, kind="stable").index).head(cfg.max_candidates)
    return df.reset_index(drop=True)


class Policy:
    """No rebalancing (also the base class)."""

    def __init__(self) -> None:
        self.name = "none"

    def plan(self, t0: pd.Timestamp, shift: ShiftSpec, bikes: np.ndarray) -> routing.RoutePlan | None:
        return None


BandFn = Callable[[pd.Timestamp, ShiftSpec], tuple[np.ndarray, np.ndarray]]


def hourly_band_fn(forecast: np.ndarray, test_start: pd.Timestamp) -> BandFn:
    """Band from an hourly forecast grid (S x hours since ``test_start``)."""

    def fn(t0: pd.Timestamp, shift: ShiftSpec) -> tuple[np.ndarray, np.ndarray]:
        h0 = int((t0 - test_start) / pd.Timedelta(hours=1))
        return path_band(forecast[:, h0 : h0 + (shift[3] - shift[1])])

    return fn


def table_band_fn(table: pd.DataFrame, pred: pd.DataFrame, n_stations: int) -> BandFn:
    """Band looked up from a station x date x shift prediction table."""
    lookup = {}
    for (date, shift), g in pred.join(table[["station", "date", "shift"]]).groupby(["date", "shift"]):
        d = np.zeros(n_stations)
        f = np.zeros(n_stations)
        d[g["station"].to_numpy()] = g["drain"].to_numpy()
        f[g["station"].to_numpy()] = g["fill"].to_numpy()
        lookup[(pd.Timestamp(date), shift)] = (d, f)

    def fn(t0: pd.Timestamp, shift: ShiftSpec) -> tuple[np.ndarray, np.ndarray]:
        return lookup[(t0.normalize(), shift[0])]

    return fn


@dataclass
class ForecastPolicy(Policy):
    """Re-plan every shift from live inventory + a forecast of each station's drain / fill band."""

    name: str
    stations: pd.DataFrame
    band_fn: BandFn
    depot: tuple[float, float]
    cfg: FleetConfig
    checks: list = field(default_factory=list)

    def plan(self, t0: pd.Timestamp, shift: ShiftSpec, bikes: np.ndarray) -> routing.RoutePlan:
        length_min = shift[2]
        drain, fill = self.band_fn(t0, shift)
        cap = self.stations["capacity"].to_numpy()
        _, move = target_levels(bikes, cap, drain, fill, self.cfg.buffer)
        plan = routing.solve(self.depot, candidates(self.stations, move, self.cfg), self.cfg, length_min * 60)
        self.checks.append(
            {"shift": shift[0], "time": str(t0), **routing.check_plan(plan, self.cfg, length_min * 60), "km": plan.distance_m / 1000}
        )
        return plan


@dataclass
class FixedRoutePolicy(Policy):
    """Status quo: the same routes every day, planned once from training-period weekday averages.

    Stops and planned volumes come from each station's chronic imbalance: overnight trucks undo the
    average weekday net flow of a whole day, midday trucks undo the average 00-10 h accumulation.
    At each stop the driver levels the dock to a fixed target (the midpoint of its average-weekday
    band for the coming horizon) within truck limits; they can see the dock, but not a forecast.
    """

    name: str
    plans: dict[str, routing.RoutePlan]

    @classmethod
    def build(cls, stations: pd.DataFrame, weekday_profile: np.ndarray, depot: tuple[float, float], cfg: FleetConfig) -> FixedRoutePolicy:
        cap = stations["capacity"].to_numpy()
        plans = {}
        for name, start_h, length_min, end_h in cfg.shifts:
            drain, fill = path_band(weekday_profile[:, start_h:end_h])
            tgt = np.clip(np.round((cfg.buffer + drain + cap - cfg.buffer - fill) / 2), 0, cap).astype(int)
            backlog = weekday_profile.sum(axis=1) if start_h == 0 else weekday_profile[:, :start_h].sum(axis=1)
            move = -np.round(backlog).astype(int)
            cand = candidates(stations, move, cfg, target=tgt)
            plan = routing.solve(depot, cand, cfg, length_min * 60, solution_limit=2000)
            tmap = dict(zip(cand["station"], cand["target"], strict=True))
            for stops in plan.routes:
                for s in stops:
                    s.target = int(tmap[s.station])
            plans[name] = plan
        return cls(name="fixed_routes", plans=plans)

    def plan(self, t0: pd.Timestamp, shift: ShiftSpec, bikes: np.ndarray) -> routing.RoutePlan:
        return self.plans[shift[0]]


@dataclass
class SimResult:
    policy: str
    empty_min: np.ndarray
    full_min: np.ndarray
    lost: np.ndarray
    diverted: np.ndarray
    bikes_moved: int
    stops_made: int
    truck_km: float
    leftover_on_trucks: int
    dropped_at_route_end: int = 0
    plans: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "empty_dock_hours": round(float(self.empty_min.sum() / 60), 1),
            "full_dock_hours": round(float(self.full_min.sum() / 60), 1),
            "stockout_hours": round(float((self.empty_min.sum() + self.full_min.sum()) / 60), 1),
            "lost_departures": int(self.lost.sum()),
            "diverted_arrivals": int(self.diverted.sum()),
            "bikes_moved": int(self.bikes_moved),
            "stops_made": int(self.stops_made),
            "truck_km": round(float(self.truck_km), 1),
            "bikes_left_on_trucks": int(self.leftover_on_trucks),
            "bikes_dropped_at_route_end": int(self.dropped_at_route_end),
        }


def simulate(
    stations: pd.DataFrame,
    trips: pd.DataFrame,
    policy: Policy,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: FleetConfig,
    init_fill: float = 0.5,
    n_alt: int = 12,
) -> SimResult:
    cap = stations["capacity"].to_numpy(int)
    S = len(cap)
    pos = pd.Series(np.arange(S), index=stations["station_id"].astype(str).to_numpy())
    tr = trips[(trips["start_time"] >= start) & (trips["start_time"] < end)]
    s_idx = tr["start_id"].map(pos).to_numpy()
    e_idx = tr["end_id"].map(pos).to_numpy()
    ts = ((tr["start_time"] - start).dt.total_seconds()).to_numpy(np.int64)
    te = ((tr["end_time"] - start).dt.total_seconds()).to_numpy(np.int64)
    horizon = int((end - start).total_seconds())
    n = len(tr)
    # trip events: (time, kind 1 = depart / 2 = arrive, station, ride)
    ev_t = np.r_[ts, te]
    ev_k = np.r_[np.ones(n, int), np.full(n, 2)]
    ev_s = np.r_[s_idx, e_idx].astype(int)
    ev_r = np.r_[np.arange(n), np.arange(n)]
    order = np.lexsort((ev_k, ev_t))
    ev_t, ev_k, ev_s, ev_r = ev_t[order], ev_k[order], ev_s[order], ev_r[order]

    lat, lon = stations["lat"].to_numpy(), stations["lon"].to_numpy()
    from .geo import haversine_matrix

    alt = np.argsort(haversine_matrix(lat, lon, lat, lon), axis=1)[:, 1 : n_alt + 1]

    bikes = np.round(cap * init_fill).astype(int)
    last = np.zeros(S, dtype=np.int64)
    empty = np.zeros(S, dtype=np.int64)
    full = np.zeros(S, dtype=np.int64)
    lost = np.zeros(S, dtype=np.int64)
    diverted = np.zeros(S, dtype=np.int64)
    cancelled = np.zeros(n, dtype=bool)

    def touch(s: int, t: int) -> None:
        if bikes[s] == 0:
            empty[s] += t - last[s]
        elif bikes[s] >= cap[s]:
            full[s] += t - last[s]
        last[s] = t

    # scheduled operations: (time, seq, kind, payload). kind 0 = plan a shift, 1 = truck stop
    heap: list = []
    seq = 0
    for day in pd.date_range(start, end, freq="D", inclusive="left"):
        for shift in cfg.shifts:
            t0 = day + pd.Timedelta(hours=shift[1])
            heapq.heappush(heap, (int((t0 - start).total_seconds()), seq, 0, (t0, shift)))
            seq += 1
    loads: dict[tuple[int, int], int] = {}
    moved = stops_made = dropped_at_end = 0
    km = 0.0
    plans = []

    i = 0
    while i < len(ev_t) or heap:
        if heap and (i >= len(ev_t) or heap[0][0] <= ev_t[i]):
            t, _, kind, payload = heapq.heappop(heap)
            if t >= horizon:
                continue
            if kind == 0:
                t0, shift = payload
                plan = policy.plan(t0, shift, bikes.copy())
                if plan is None:
                    continue
                plans.append(plan)
                km += plan.distance_m / 1000
                pid = len(plans)
                for v, stops in enumerate(plan.routes):
                    loads[(pid, v)] = 0
                    later = np.cumsum([max(0, s.move) for s in stops][::-1])[::-1]
                    for j, s in enumerate(stops):
                        still_to_deliver = int(later[j + 1]) if j + 1 < len(stops) else 0
                        heapq.heappush(heap, (t + s.eta_s, seq, 1, (pid, v, s, still_to_deliver, j == len(stops) - 1)))
                        seq += 1
            else:
                pid, v, stop, still_to_deliver, is_last = payload
                st = stop.station
                load = loads[(pid, v)]
                want = stop.move if stop.target is None else stop.target - bikes[st]
                if want > 0:
                    q = min(want, load, cap[st] - bikes[st])
                else:
                    # never pick up more than the rest of the route plans to drop off
                    q = -max(0, min(-want, bikes[st], cfg.truck_capacity - load, still_to_deliver - load))
                if q != 0:
                    touch(st, t)
                    bikes[st] += q
                    loads[(pid, v)] = load - q
                    moved += abs(q)
                stops_made += 1
                if is_last and loads[(pid, v)] > 0:
                    # end of route: drop any bikes still on board at the nearest docks with space
                    for a in np.r_[st, alt[st]]:
                        free = int(cap[a] - bikes[a])
                        q = min(free, loads[(pid, v)])
                        if q > 0:
                            touch(a, t)
                            bikes[a] += q
                            loads[(pid, v)] -= q
                            moved += q
                            dropped_at_end += q
                        if loads[(pid, v)] == 0:
                            break
            continue
        t, k, s, r = int(ev_t[i]), ev_k[i], ev_s[i], ev_r[i]
        i += 1
        if t >= horizon:
            continue
        if k == 1:
            if bikes[s] > 0:
                touch(s, t)
                bikes[s] -= 1
            else:
                lost[s] += 1
                cancelled[r] = True
        else:
            if cancelled[r]:
                continue
            if bikes[s] < cap[s]:
                touch(s, t)
                bikes[s] += 1
            else:
                diverted[s] += 1
                for a in alt[s]:
                    if bikes[a] < cap[a]:
                        touch(a, t)
                        bikes[a] += 1
                        break
    for s in range(S):
        touch(s, horizon)
    return SimResult(
        policy=policy.name,
        empty_min=empty / 60.0,
        full_min=full / 60.0,
        lost=lost,
        diverted=diverted,
        bikes_moved=moved,
        stops_made=stops_made,
        truck_km=km,
        leftover_on_trucks=int(sum(loads.values())),
        dropped_at_route_end=int(dropped_at_end),
        plans=plans,
    )
