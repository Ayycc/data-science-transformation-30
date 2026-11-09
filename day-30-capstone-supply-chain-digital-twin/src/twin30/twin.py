"""SimPy digital twin of the supplier -> DC -> store network.

Each replication draws one posterior sample of the demand model (parameter uncertainty), daily
negative-binomial demand (noise), bootstrapped supplier lead times, outage windows (random from the
calibrated rate, replayed from the log, or forced by a stress scenario) and late trucks. The
replenishment rules are the retailer's own (``policy.py``).

Event timing within a day matches the history: deliveries land at the start of day t (integer
time t), trading and ordering happen at t + 0.5, DCs order from suppliers on Mondays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import simpy

from . import policy as pol
from .calibrate import Calibration, sample_outages
from .config import FINANCE, HISTORICAL_ASSIGNMENT, N_CATS, N_DCS, N_STORES, N_SUPPLIERS, Decisions, geometry
from .demand import Posterior
from .kpis import Ledger, compute


@dataclass
class StartState:
    store_on_hand: np.ndarray  # (S, C)
    dc_on_hand: np.ndarray  # (D, C)
    store_pipeline: list  # [(rel_day, (S, C) array)]
    dc_pipeline: list  # [(rel_day, supplier, dc, (C,) units)]
    sales_hist: np.ndarray  # (W, S, C)
    outflow_hist: np.ndarray  # (W, D, C)

    @classmethod
    def from_snapshot(cls, snap: dict) -> StartState:
        return cls(
            store_on_hand=np.array(snap["store_on_hand"], dtype=float),
            dc_on_hand=np.array(snap["dc_on_hand"], dtype=float),
            store_pipeline=[(int(d), np.array(v, dtype=float)) for d, v in snap["store_pipeline"]],
            dc_pipeline=[(int(r), int(k), int(d), np.array(v, dtype=float)) for r, k, d, v in snap["dc_pipeline"]],
            sales_hist=np.array(snap["sales_hist"], dtype=float),
            outflow_hist=np.array(snap["outflow_hist"], dtype=float),
        )


@dataclass
class Scenario:
    """Everything one simulation needs apart from the random seed."""

    decisions: Decisions
    day0: int  # absolute day index of the first simulated day (for calendar and trend)
    horizon: int
    promo: np.ndarray  # (H, C)
    post: np.ndarray  # (H, C)
    outage_mode: str = "random"  # random | replay | none
    replay_outages: list = field(default_factory=list)  # [(supplier, start_rel, end_rel)]
    forced_outages: list = field(default_factory=list)  # added on top of random/replay


class TwinSim:
    def __init__(
        self,
        cal: Calibration,
        posterior: Posterior,
        scen: Scenario,
        start: StartState,
        rng: np.random.Generator,
        draw: int,
    ) -> None:
        self.cal, self.post, self.scen, self.rng = cal, posterior, scen, rng
        dec = scen.decisions
        self.policy = dec.policy()
        self.assign = np.array(dec.assignment, dtype=int)
        self.open = np.zeros(N_DCS, dtype=bool)
        self.open[np.unique(self.assign)] = True
        geo = geometry()
        self.transit = geo.transit[self.assign, np.arange(N_STORES)]
        self.cover = self.transit + 1
        self.cap = np.array(FINANCE.dc_capacity_units_per_day)
        H = scen.horizon
        mu = posterior.mean(draw, scen.day0, H, scen.promo, scen.post)
        self.demand = posterior.sample_demand(draw, mu, rng)
        # random draws are taken in a fixed order so decisions share common random numbers
        self.delay = rng.random((H, N_STORES)) < cal.transit_delay_p
        self.lead_u = rng.random((H, N_DCS, N_SUPPLIERS))
        if scen.outage_mode == "random":
            outages = sample_outages(cal, H, rng)
        elif scen.outage_mode == "replay":
            outages = list(scen.replay_outages)
        else:
            outages = []
        self.outages = outages + list(scen.forced_outages)
        self._lead_sorted = [np.sort(np.asarray(x if len(x) else [1])) for x in cal.lead_samples]

        # state
        self.store_oh = start.store_on_hand.copy()
        self.dc_oh = self._initial_dc_stock(start)
        self.in_transit = np.zeros((N_STORES, N_CATS))
        self.on_order = np.zeros((N_DCS, N_CATS))
        W = self.policy.forecast_window
        self.sales_hist = np.concatenate([start.sales_hist[-W:], np.zeros((H, N_STORES, N_CATS))])
        self.req_hist = np.concatenate([self._initial_outflow(start)[-W:], np.zeros((H, N_DCS, N_CATS))])
        self.W = W
        # records
        self.sales = np.zeros((H, N_STORES, N_CATS))
        self.stockout = np.zeros((H, N_STORES, N_CATS))
        self.rec_store = np.zeros((H, N_STORES, N_CATS))
        self.rec_dc = np.zeros((H, N_DCS, N_CATS))
        self.ship = np.zeros((H, N_STORES, N_CATS))
        self.receipts = np.zeros((N_SUPPLIERS, N_CATS))
        self.env = simpy.Environment()
        for d, v in start.store_pipeline:
            self.in_transit += v
            self.env.process(self._deliver_store(max(d, 0), v))
        for r, k, d, q in start.dc_pipeline:
            v = np.zeros((N_DCS, N_CATS))
            v[d] = q
            v = self._remap_dc(v)
            self.on_order += v
            self.env.process(self._deliver_dc(max(r, 0), v, k))
        self.env.process(self._trading_day())

    # --- initial conditions -------------------------------------------------------------------
    def _initial_dc_stock(self, start: StartState) -> np.ndarray:
        oh = start.dc_on_hand.copy()
        if not self.open[2] or oh[2].sum() > 0:
            return oh
        # opening DC-East: pre-stock it from the other DCs in proportion to the demand it takes over
        share = np.zeros((N_DCS, N_CATS))
        rate = start.sales_hist.mean(axis=0)
        for d in range(N_DCS):
            share[d] = rate[self.assign == d].sum(axis=0)
        share /= share.sum(axis=0, keepdims=True)
        return oh.sum(axis=0, keepdims=True) * share

    def _remap_dc(self, v: np.ndarray) -> np.ndarray:
        """Inbound POs already placed keep their DC unless that DC is closed in this scenario."""
        v = v.copy()
        for d in range(N_DCS):
            if not self.open[d] and v[d].sum() > 0:
                v[0] += v[d]
                v[d] = 0
        return v

    def _initial_outflow(self, start: StartState) -> np.ndarray:
        """Trailing store-request history per DC. Kept as recorded when the network is unchanged;
        otherwise re-aggregated from store sales by the new assignment, scaled to the recorded
        request/sales ratio of each category."""
        hist = start.sales_hist
        rec = start.outflow_hist
        if rec.shape[1:] == (N_DCS, N_CATS) and np.array_equal(self.assign, np.array(HISTORICAL_ASSIGNMENT)):
            return rec.copy()
        ratio = rec.sum(axis=(0, 1)) / np.maximum(hist.sum(axis=(0, 1)), 1e-9)
        out = np.zeros((hist.shape[0], N_DCS, N_CATS))
        for d in range(N_DCS):
            out[:, d] = hist[:, self.assign == d].sum(axis=1) * ratio[None, :]
        return out

    # --- processes ----------------------------------------------------------------------------
    def _deliver_store(self, day: int, v: np.ndarray):
        yield self.env.timeout(day - self.env.now)
        self.store_oh += v
        self.in_transit -= v

    def _deliver_dc(self, day: int, v: np.ndarray, supplier: int | None):
        yield self.env.timeout(day - self.env.now)
        self.dc_oh += v
        self.on_order -= v
        if supplier is not None:
            self.receipts[supplier] += v.sum(axis=0)

    def _dispatch_day(self, k: int, t: int) -> int:
        d = t
        for kk, st, en in self.outages:
            if kk == k and st <= d < en:
                d = en
        return d

    def _lead(self, k: int, u: float) -> int:
        samples = self._lead_sorted[k]
        return int(samples[min(int(u * len(samples)), len(samples) - 1)])

    def _trading_day(self):
        yield self.env.timeout(0.5)
        W, sc = self.W, self.scen
        for t in range(sc.horizon):
            dem = self.demand[t]
            s = np.minimum(dem, self.store_oh)
            self.stockout[t] = dem > self.store_oh
            self.store_oh = self.store_oh - s
            self.sales[t] = s
            self.sales_hist[W + t] = s
            soon = pol.promo_in_window(sc.promo, t, self.cover)
            req = pol.store_orders(self.store_oh, self.in_transit, self.sales_hist[t + 1 : W + t + 1], self.cover, soon, self.policy)
            ship = pol.ration(req, self.assign, self.dc_oh, self.cap)
            for d in range(N_DCS):
                rows = self.assign == d
                self.dc_oh[d] -= ship[rows].sum(axis=0)
                self.req_hist[W + t, d] = req[rows].sum(axis=0)
            self.ship[t] = ship
            self.in_transit += ship
            arrive = t + self.transit + self.delay[t].astype(int)
            for a in np.unique(arrive):
                v = np.where((arrive == a)[:, None], ship, 0.0)
                if v.sum() > 0:
                    self.env.process(self._deliver_store(int(a), v))
            if (sc.day0 + t) % 7 == 0:
                orders = pol.dc_orders(self.dc_oh, self.on_order, self.req_hist[t + 1 : W + t + 1], self.policy)
                for d in range(N_DCS):
                    if not self.open[d]:
                        continue
                    for k in range(N_SUPPLIERS):
                        q = orders[d, :, k]
                        if q.sum() <= 0:
                            continue
                        receipt = self._dispatch_day(k, t) + self._lead(k, self.lead_u[t, d, k])
                        v = np.zeros((N_DCS, N_CATS))
                        v[d] = q
                        self.on_order += v
                        self.env.process(self._deliver_dc(receipt, v, k))
            self.rec_store[t] = self.store_oh
            self.rec_dc[t] = self.dc_oh
            yield self.env.timeout(1)

    def run(self) -> Ledger:
        self.env.run(until=self.scen.horizon)
        return Ledger(
            sales=self.sales,
            stockout=self.stockout,
            store_oh=self.rec_store,
            dc_oh=self.rec_dc,
            ship=self.ship,
            assign=self.assign,
            promo=self.scen.promo,
            receipts=self.receipts,
            open_dcs=self.open,
            demand=self.demand,
        )


def simulate(
    cal: Calibration, posterior: Posterior, scen: Scenario, start: StartState, seed: int, draw: int | None = None
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draw = int(seed % posterior.n) if draw is None else draw
    return compute(TwinSim(cal, posterior, scen, start, rng, draw).run())
