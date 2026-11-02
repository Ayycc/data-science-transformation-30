"""Seeded SYNTHETIC telemetry generator for a 20-service platform.

Per service and minute it emits latency (ms), error rate, CPU (%) and queue depth (in-flight requests).
Faults injected at a root service propagate to every transitive caller through the call graph
(caller latency includes callee latency, caller errors include un-retried callee errors) with one minute
of lag per hop. Benign deploys, single-metric blips and platform-wide traffic surges add realistic noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .topology import SERVICES, call_graph, callees_first_order, upstream_depths

METRICS: list[str] = ["latency_ms", "error_rate", "cpu_pct", "queue_depth"]
FAULTS: list[str] = ["cpu_saturation", "error_burst", "latency_regression", "queue_backlog", "slow_leak"]
MIN_PER_DAY = 1440
STORES = ("cache", "catalog-db", "order-db", "user-db", "message-queue")


@dataclass(frozen=True)
class SimConfig:
    n_days: int = 16
    train_days: int = 2
    n_incidents: int = 60
    seed: int = 20261102
    benign_deploys_per_day: float = 8.0
    p_bad_deploy: float = 0.55  # share of incidents caused by a deploy on the root service
    p_decoy_deploy: float = 0.30  # share of incidents with an unrelated deploy on an affected caller
    blips_per_day: float = 14.0
    n_surges: int = 3
    n_concurrent: int = 8  # incidents that start 0-10 min after another, unrelated one

    @property
    def n_minutes(self) -> int:
        return self.n_days * MIN_PER_DAY


@dataclass
class Telemetry:
    values: np.ndarray  # (T, S, M) float32
    services: list[str]
    metrics: list[str]
    incidents: pd.DataFrame
    deploys: pd.DataFrame
    surges: pd.DataFrame
    config: SimConfig

    @property
    def n_minutes(self) -> int:
        return self.values.shape[0]


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    eps = rng.normal(0.0, sd * np.sqrt(1 - phi**2), n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def _profile(n: int, start: int, dur: int, ramp: int, kind: str) -> np.ndarray:
    """Fault intensity in [0, 1] over time."""
    p = np.zeros(n)
    t = np.arange(n)
    if kind == "slow_leak":
        on = (t >= start) & (t < start + dur)
        p[on] = (t[on] - start + 1) / dur
    else:
        on = (t >= start) & (t < start + dur)
        p[on] = np.minimum(1.0, (t[on] - start + 1) / ramp)
    rec = (t >= start + dur) & (t < start + dur + 5)
    p[rec] = 1.0 - (t[rec] - (start + dur) + 1) / 6.0
    return p


def _shift1(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[:1], x[:-1]])


def _schedule_incidents(cfg: SimConfig, rng: np.random.Generator) -> list[dict]:
    """Incidents in non-overlapping slots after the training window. Roots are drawn with probability
    proportional to sqrt(blast radius), so deeper services (larger blast radius) fail more often. In
    `n_concurrent` slots a second, unrelated incident (neither root upstream of the other) starts 0-10
    minutes after the first, which is the case that topological grouping has to separate."""
    g = call_graph()
    first = cfg.train_days * MIN_PER_DAY
    n_slots = cfg.n_incidents - cfg.n_concurrent
    slot = (cfg.n_minutes - first) // n_slots
    radius = np.array([len(upstream_depths(g, s)) for s in SERVICES], dtype=float)
    p_root = np.sqrt(radius) / np.sqrt(radius).sum()
    concurrent = set(rng.choice(n_slots, cfg.n_concurrent, replace=False).tolist())
    rows: list[dict] = []

    def make(root: str, start: int) -> dict:
        fault = FAULTS[rng.integers(len(FAULTS))]
        if root in STORES and rng.random() < 0.5:
            fault = "queue_backlog"
        dur = int(rng.integers(20, 61))
        depths = upstream_depths(g, root)
        return dict(
            root=root,
            fault=fault,
            start=start,
            end=start + dur,
            severity=float(rng.uniform(0.6, 1.4)),
            n_affected=len(depths),
            affected=";".join(sorted(depths, key=lambda s: (depths[s], s))),
            depths=depths,
        )

    for i in range(n_slots):
        root = SERVICES[rng.choice(len(SERVICES), p=p_root)]
        start = int(first + i * slot + rng.integers(60, slot - 140))
        rows.append(make(root, start))
        if i in concurrent:
            anc = set(upstream_depths(g, root))
            ok = [s for s in SERVICES if s not in anc and root not in upstream_depths(g, s)]
            w = p_root[[SERVICES.index(s) for s in ok]]
            r2 = ok[rng.choice(len(ok), p=w / w.sum())]
            rows.append(make(r2, start + int(rng.integers(0, 11))) | {"concurrent": True})
            rows[-2]["concurrent"] = True
    for k, r in enumerate(rows):
        r["incident_id"] = k
        r.setdefault("concurrent", False)
    return rows


def generate(cfg: SimConfig | None = None) -> Telemetry:
    """Generate the full SYNTHETIC telemetry cube, incident ledger and deploy log."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    g = call_graph()
    n, S = cfg.n_minutes, len(SERVICES)
    idx = {s: i for i, s in enumerate(SERVICES)}
    t = np.arange(n)

    # Shared traffic: diurnal + weekly + slow AR noise + small day-to-day level change
    tod = (t % MIN_PER_DAY) / MIN_PER_DAY
    day_level = np.repeat(1 + rng.normal(0, 0.02, cfg.n_days), MIN_PER_DAY)
    traffic = day_level * (1 + 0.40 * np.sin(2 * np.pi * (tod - 0.30)) + 0.08 * np.sin(4 * np.pi * (tod - 0.10))) + _ar1(rng, n, 0.98, 0.03)
    surges = []
    first = cfg.train_days * MIN_PER_DAY
    for _ in range(cfg.n_surges):
        s0 = int(rng.integers(first, n - 120))
        traffic = traffic * (1 + 0.35 * _profile(n, s0, 40, 8, "step"))
        surges.append(dict(start=s0, end=s0 + 40))

    # Service parameters
    is_store = {s: s in STORES for s in SERVICES}
    own_lat = {s: (rng.uniform(2, 8) if is_store[s] else rng.uniform(8, 30)) for s in SERVICES}
    cpu_base = {s: rng.uniform(18, 40) for s in SERVICES}
    err_base = {s: rng.uniform(0.001, 0.006) for s in SERVICES}
    req_rate = {s: rng.uniform(40, 200) for s in SERVICES}  # req/s at traffic 1
    w_lat = {e: rng.uniform(0.5, 1.0) for e in g.edges}
    p_err = {e: rng.uniform(0.5, 0.8) for e in g.edges}

    lat_mult = np.ones((n, S))
    err_add = np.zeros((n, S))
    cpu_add = np.zeros((n, S))
    q_mult = np.ones((n, S))

    incidents = _schedule_incidents(cfg, rng)
    deploys: list[dict] = []
    for inc in incidents:
        j = idx[inc["root"]]
        sev = inc["severity"]
        p = _profile(n, inc["start"], inc["end"] - inc["start"], 3, "slow_leak" if inc["fault"] == "slow_leak" else "step")
        f = inc["fault"]
        if f == "cpu_saturation":
            cpu_add[:, j] += 45 * sev * p
            lat_mult[:, j] *= 1 + 2.0 * sev * p
        elif f == "error_burst":
            err_add[:, j] += 0.10 * sev * p
            lat_mult[:, j] *= 1 + 0.3 * sev * p
        elif f == "latency_regression":
            lat_mult[:, j] *= 1 + 3.0 * sev * p
        elif f == "queue_backlog":
            q_mult[:, j] *= 1 + 5.0 * sev * p
            lat_mult[:, j] *= 1 + 1.5 * sev * p
        elif f == "slow_leak":
            cpu_add[:, j] += 40 * sev * p
            lat_mult[:, j] *= 1 + 2.5 * sev * p
        inc["deploy_on_root"] = bool(rng.random() < cfg.p_bad_deploy)
        if inc["deploy_on_root"]:
            deploys.append(dict(minute=int(inc["start"] - rng.integers(2, 13)), service=inc["root"], kind="bad", incident_id=inc["incident_id"]))
        callers = [s for s in inc["depths"] if s != inc["root"]]
        inc["decoy_deploy"] = bool(callers and rng.random() < cfg.p_decoy_deploy)
        if inc["decoy_deploy"]:
            deploys.append(
                dict(
                    minute=int(inc["start"] + rng.integers(-15, 16)),
                    service=callers[rng.integers(len(callers))],
                    kind="decoy",
                    incident_id=inc["incident_id"],
                )
            )

    n_benign = int(cfg.benign_deploys_per_day * cfg.n_days)
    for _ in range(n_benign):
        deploys.append(dict(minute=int(rng.integers(10, n - 10)), service=SERVICES[rng.integers(S)], kind="benign", incident_id=-1))
    # Every deploy restarts pods: a 2-minute CPU/latency blip on that service
    for d in deploys:
        j = idx[d["service"]]
        m = d["minute"]
        cpu_add[m : m + 2, j] += 20
        lat_mult[m : m + 2, j] *= 1.4

    # Propagate callees -> callers (noise-free state), one minute lag per hop
    lat = np.zeros((n, S))
    err = np.zeros((n, S))
    for s in callees_first_order(g):
        j = idx[s]
        tr = traffic * (1 + _ar1(rng, n, 0.9, 0.02))
        lat_s = own_lat[s] * (1 + 0.25 * (tr - 1)) * lat_mult[:, j]
        keep = 1 - np.clip(err_base[s] * (1 + 0.3 * (tr - 1)) + err_add[:, j], 0, 1)
        retry_cpu = np.zeros(n)
        for c in g.successors(s):
            jc = idx[c]
            lat_s = lat_s + w_lat[(s, c)] * _shift1(lat[:, jc])
            keep = keep * (1 - p_err[(s, c)] * _shift1(err[:, jc]))
            retry_cpu += 60 * _shift1(err[:, jc])
        lat[:, j] = lat_s
        err[:, j] = 1 - keep
        cpu_add[:, j] += retry_cpu
        cpu_add[:, j] += cpu_base[s] * (tr - 1)  # load-proportional CPU

    values = np.zeros((n, S, len(METRICS)), dtype=np.float32)
    for s in SERVICES:
        j = idx[s]
        rps = req_rate[s] * traffic
        n_req = np.maximum(1, rng.poisson(rps * 60))
        values[:, j, 0] = lat[:, j] * np.exp(rng.normal(0, 0.06, n))
        values[:, j, 1] = rng.binomial(n_req, np.clip(err[:, j], 0, 1)) / n_req
        values[:, j, 2] = np.clip(cpu_base[s] + cpu_add[:, j] + _ar1(rng, n, 0.8, 2.0), 0, 100)
        inflight = rps * lat[:, j] / 1000.0 * q_mult[:, j]
        values[:, j, 3] = rng.poisson(np.maximum(inflight, 0.05) * 4) / 4.0

    # Single-metric blips (e.g. GC pause, scrape glitch): 1-3 minutes
    n_blips = int(cfg.blips_per_day * cfg.n_days)
    blip_rows = []
    for _ in range(n_blips):
        j, m = int(rng.integers(S)), int(rng.integers(len(METRICS)))
        t0, dur = int(rng.integers(0, n - 4)), int(rng.integers(1, 4))
        factor = {0: 2.5, 1: None, 2: None, 3: 3.0}[m]
        if m == 1:
            values[t0 : t0 + dur, j, m] += 0.03
        elif m == 2:
            values[t0 : t0 + dur, j, m] = np.minimum(100, values[t0 : t0 + dur, j, m] + 30)
        else:
            values[t0 : t0 + dur, j, m] *= factor
        blip_rows.append((t0, SERVICES[j], METRICS[m]))

    inc_df = pd.DataFrame([{k: v for k, v in r.items() if k != "depths"} for r in incidents])
    inc_df["true_onsets"] = [";".join(f"{s}:{r['start'] + d}" for s, d in sorted(r["depths"].items(), key=lambda kv: kv[1])) for r in incidents]
    dep_df = pd.DataFrame(deploys).sort_values("minute").reset_index(drop=True)
    return Telemetry(values, list(SERVICES), list(METRICS), inc_df, dep_df, pd.DataFrame(surges), cfg)


def to_wide_frame(tel: Telemetry, start: int, stop: int) -> pd.DataFrame:
    """Wide frame (minute x service__metric) for a time slice, used for the committed sample."""
    block = tel.values[start:stop].reshape(stop - start, -1)
    cols = [f"{s}__{m}" for s in tel.services for m in tel.metrics]
    df = pd.DataFrame(block, columns=cols)
    df.insert(0, "minute", np.arange(start, stop))
    return df
