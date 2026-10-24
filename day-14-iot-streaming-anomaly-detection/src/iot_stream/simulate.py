"""SYNTHETIC manufacturing IoT stream: run-to-failure trajectories, drift, dropouts, late data.

Every number here is simulated from a fixed seed. Nothing is a real factory measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DAY, HOUR, SimConfig


@dataclass
class SensorStream:
    """Events in ARRIVAL order (what a consumer sees), plus ground truth for evaluation."""

    arrival: np.ndarray  # float64 seconds since stream start
    event_time: np.ndarray  # float64 seconds since stream start
    machine: np.ndarray  # int32
    sensor: np.ndarray  # int8 (index into config.SENSORS)
    value: np.ndarray  # float32
    failures: pd.DataFrame  # machine, failure_time, onset_time, lifecycle_start, repair_end, sudden
    horizon_s: float
    n_machines: int
    meta: dict

    def __len__(self) -> int:
        return int(self.arrival.shape[0])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "arrival": self.arrival,
                "event_time": self.event_time,
                "machine": self.machine,
                "sensor": self.sensor,
                "value": self.value,
            }
        )


def _lifecycles(cfg: SimConfig, rng: np.random.Generator, t_grid: np.ndarray, horizon: float):
    """Per machine: sequence of lifecycles ending in failure (or censored at horizon)."""
    m_count, n_ticks = cfg.n_machines, t_grid.shape[0]
    deg_v = np.zeros((m_count, n_ticks))
    deg_t = np.zeros((m_count, n_ticks))
    deg_c = np.zeros((m_count, n_ticks))
    lc_mult = np.ones((m_count, n_ticks))
    up = np.ones((m_count, n_ticks), dtype=bool)
    rows = []
    for m in range(m_count):
        start = 0.0
        first = True
        while start < horizon:
            life = rng.weibull(cfg.weibull_shape) * cfg.weibull_scale_days * DAY
            if first:
                life *= rng.uniform(0.05, 1.0)  # machines enter the window part-way through their life
                first = False
            fail = start + max(life, 12 * HOUR)
            sudden = bool(rng.random() < cfg.sudden_failure_share)
            dur = rng.uniform(*cfg.degradation_hours) * HOUR
            onset = max(fail - dur, start + 6 * HOUR)
            power = rng.uniform(1.5, 3.0)
            av, at, ac = rng.uniform(0.8, 2.5), rng.uniform(4.0, 12.0), rng.uniform(0.05, 0.25)
            idx = (t_grid >= start) & (t_grid < fail)
            if not sudden:
                h = np.clip((t_grid[idx] - onset) / (fail - onset), 0.0, 1.0) ** power
                deg_v[m, idx] = av * h
                deg_t[m, idx] = at * h
                deg_c[m, idx] = ac * h
            lc_mult[m, idx] = rng.normal(1.0, 0.04)
            if fail >= horizon:
                break
            repair = fail + rng.uniform(*cfg.downtime_hours) * HOUR
            up[m, (t_grid >= fail) & (t_grid < repair)] = False
            rows.append(
                {
                    "machine": m,
                    "lifecycle_start": start,
                    "onset_time": onset if not sudden else np.nan,
                    "failure_time": fail,
                    "repair_end": min(repair, horizon),
                    "sudden": sudden,
                }
            )
            start = repair
    failures = pd.DataFrame(rows, columns=["machine", "lifecycle_start", "onset_time", "failure_time", "repair_end", "sudden"])
    return deg_v, deg_t, deg_c, lc_mult, up, failures


def simulate(cfg: SimConfig | None = None) -> SensorStream:
    """Generate the SYNTHETIC fleet stream in arrival order."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    n_ticks = int(round(cfg.days * DAY / cfg.tick_s))
    t_grid = np.arange(n_ticks) * cfg.tick_s
    horizon = n_ticks * cfg.tick_s
    m_count = cfg.n_machines

    b_v = rng.normal(2.0, 0.3, m_count).clip(1.0)[:, None]
    b_t = rng.normal(55.0, 3.0, m_count)[:, None]
    b_c = rng.normal(20.0, 2.0, m_count).clip(8.0)[:, None]
    load_phase = rng.uniform(0, DAY, m_count)[:, None]
    load_amp = rng.uniform(0.03, 0.08, m_count)[:, None]

    deg_v, deg_t, deg_c, lc_mult, up, failures = _lifecycles(cfg, rng, t_grid, horizon)

    # per-machine 3-shift production schedule with its own phase (8 h blocks)
    shift = np.floor(((t_grid[None, :] + load_phase) % DAY) / (8 * HOUR)).astype(int)
    load = 1.0 + load_amp * np.array([0.0, 1.0, -1.0])[shift]
    regime = t_grid[None, :] >= cfg.regime_change_day * DAY
    vib_mult = np.where(regime, cfg.regime_vib_mult, 1.0)
    cur_mult = np.where(regime, cfg.regime_cur_mult, 1.0)
    temp_add = np.where(regime, cfg.regime_temp_add, 0.0)
    warming = cfg.seasonal_warming_c * t_grid[None, :] / horizon
    calib = (rng.random(m_count) < cfg.calib_drift_share)[:, None] * cfg.calib_drift_c_per_day * t_grid[None, :] / DAY

    shape = (m_count, n_ticks)
    vib = b_v * lc_mult * load * vib_mult * (1 + deg_v) + rng.normal(0, 1, shape) * 0.08 * b_v
    temp = b_t + 25.0 * (load - 1.0) + warming + temp_add + calib + deg_t + rng.normal(0, 0.5, shape)
    cur = b_c * load * cur_mult * (1 + deg_c) + rng.normal(0, 0.25, shape)
    values = (vib, temp, cur)

    phase_off = rng.uniform(0, cfg.tick_s * 0.9, m_count)
    # gateway (store-and-forward) outages per machine: events buffered then flushed together
    outages = []
    for m in range(m_count):
        k = rng.poisson(cfg.gateway_outages_per_machine_per_day * cfg.days)
        for _ in range(k):
            g0 = rng.uniform(0, horizon)
            outages.append((m, g0, g0 + rng.uniform(*cfg.gateway_outage_minutes) * 60))

    parts = []
    mm, tt = np.meshgrid(np.arange(m_count), np.arange(n_ticks), indexing="ij")
    for s in range(3):
        keep = up.copy()
        k_drop = rng.poisson(cfg.dropouts_per_sensor_per_day * cfg.days, m_count)
        for m in np.nonzero(k_drop)[0]:
            for _ in range(k_drop[m]):
                d0 = rng.uniform(0, horizon)
                d1 = d0 + rng.uniform(*cfg.dropout_minutes) * 60
                keep[m, (t_grid >= d0) & (t_grid < d1)] = False
        keep &= rng.random(shape) >= cfg.point_loss
        m_idx, t_idx = mm[keep], tt[keep]
        et = t_grid[t_idx] + phase_off[m_idx] + s * 0.5 + np.abs(rng.normal(0, 2.0, m_idx.shape[0]))
        parts.append((m_idx.astype(np.int32), np.full(m_idx.shape[0], s, np.int8), et, values[s][keep].astype(np.float32)))

    machine = np.concatenate([p[0] for p in parts])
    sensor = np.concatenate([p[1] for p in parts])
    event_time = np.concatenate([p[2] for p in parts])
    value = np.concatenate([p[3] for p in parts])
    n = machine.shape[0]

    delay = 0.05 + rng.exponential(1.0, n)
    late = rng.random(n) < cfg.late_share
    delay[late] = rng.lognormal(np.log(120.0), 0.9, int(late.sum()))
    gateway = np.zeros(n, dtype=bool)
    by_m = np.argsort(machine, kind="stable")
    bounds = np.searchsorted(machine[by_m], np.arange(m_count + 1))
    for m, g0, g1 in outages:
        rows = by_m[bounds[m] : bounds[m + 1]]
        sel = rows[(event_time[rows] >= g0) & (event_time[rows] < g1)]
        delay[sel] = g1 - event_time[sel] + rng.uniform(0, 30, sel.shape[0])
        gateway[sel] = True
    arrival = event_time + delay
    order = np.argsort(arrival, kind="stable")

    meta = {
        "generator": "SYNTHETIC (iot_stream.simulate)",
        "seed": cfg.seed,
        "n_machines": m_count,
        "days": cfg.days,
        "tick_s": cfg.tick_s,
        "n_events": int(n),
        "n_failures": int(len(failures)),
        "n_sudden_failures": int(failures["sudden"].sum()) if len(failures) else 0,
        "share_delayed_lognormal": float(late.mean()),
        "share_gateway_buffered": float(gateway.mean()),
        "regime_change_time_s": cfg.regime_change_day * DAY,
    }
    return SensorStream(
        arrival=arrival[order],
        event_time=event_time[order],
        machine=machine[order],
        sensor=sensor[order],
        value=value[order],
        failures=failures,
        horizon_s=horizon,
        n_machines=m_count,
        meta=meta,
    )
