"""Estimate the twin's non-demand uncertainty from the observable history (lead times, outages, transit)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import DAYS_PER_YEAR, DCS, STORES, SUPPLIERS, geometry

# Weak priors for outage behaviour (a supplier with no observed outage still gets a non-zero rate)
OUTAGE_RATE_PRIOR = (1.0, 1.0)  # Gamma(shape, rate in years): prior mean 1 outage per year
OUTAGE_DURATION_PRIOR_DAYS = 10.0


@dataclass
class Calibration:
    lead_samples: list[list[int]]  # per supplier: observed dispatch->receipt days
    outage_shape: list[float]  # Gamma posterior for the Poisson outage rate (per year)
    outage_rate: list[float]
    outage_durations: list[list[int]]  # per supplier: observed outage lengths (days)
    transit_delay_p: float  # probability a DC->store truck arrives one day late
    window_days: tuple[int, int]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Calibration:
        d = dict(d)
        d["window_days"] = tuple(d["window_days"])
        return cls(**d)

    def summary(self) -> dict:
        out = {}
        for k, s in enumerate(SUPPLIERS):
            ls = np.array(self.lead_samples[k])
            out[s.name] = {
                "lead_mean_days": float(ls.mean()) if len(ls) else None,
                "lead_p90_days": float(np.percentile(ls, 90)) if len(ls) else None,
                "n_orders": int(len(ls)),
                "outages_observed": len(self.outage_durations[k]),
                "outage_rate_posterior_mean_per_year": self.outage_shape[k] / self.outage_rate[k],
                "outage_mean_days": float(np.mean(self.outage_durations[k])) if self.outage_durations[k] else None,
            }
        out["transit_delay_p"] = self.transit_delay_p
        return out


def calibrate(tabs: dict[str, pd.DataFrame], day_from: int, day_to: int) -> Calibration:
    po = tabs["purchase_orders"]
    po = po[(po.order_day >= day_from) & (po.order_day < day_to)]
    orders = po.drop_duplicates(["order_day", "supplier", "dc"])
    lead = []
    for s in SUPPLIERS:
        o = orders[orders.supplier == s.name]
        lead.append((o.receipt_day - o.dispatch_day).astype(int).tolist())
    out = tabs["outages"]
    out = out[(out.start_day >= day_from) & (out.start_day < day_to)]
    years = (day_to - day_from) / DAYS_PER_YEAR
    shape, rate, durs = [], [], []
    for s in SUPPLIERS:
        o = out[out.supplier == s.name]
        shape.append(OUTAGE_RATE_PRIOR[0] + len(o))
        rate.append(OUTAGE_RATE_PRIOR[1] + years)
        durs.append((o.end_day - o.start_day).astype(int).tolist())
    sh = tabs["shipments"]
    sh = sh[(sh.ship_day >= day_from) & (sh.ship_day < day_to)].drop_duplicates(["ship_day", "store"])
    geo = geometry()
    di = sh.dc.map({d.name: i for i, d in enumerate(DCS)}).to_numpy()
    si = sh.store.map({s.name: i for i, s in enumerate(STORES)}).to_numpy()
    extra = (sh.arrive_day - sh.ship_day).to_numpy() - geo.transit[di, si]
    return Calibration(
        lead_samples=lead,
        outage_shape=shape,
        outage_rate=rate,
        outage_durations=durs,
        transit_delay_p=float((extra > 0).mean()),
        window_days=(day_from, day_to),
    )


def sample_outages(cal: Calibration, horizon: int, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    """Random outage windows for one replication: rate drawn from its posterior, durations bootstrapped."""
    out = []
    for k in range(len(SUPPLIERS)):
        lam = rng.gamma(cal.outage_shape[k], 1.0 / cal.outage_rate[k])
        n = rng.poisson(lam * horizon / DAYS_PER_YEAR)
        pool = cal.outage_durations[k] or [OUTAGE_DURATION_PRIOR_DAYS]
        for start in rng.integers(0, horizon, size=n):
            dur = int(max(2, round(float(rng.choice(pool)) * rng.uniform(0.7, 1.3))))
            out.append((k, int(start), int(start) + dur))
    return out
