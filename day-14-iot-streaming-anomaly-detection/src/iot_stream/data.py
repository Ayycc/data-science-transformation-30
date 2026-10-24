"""Data acquisition: public NASA C-MAPSS probe (optional replay), SYNTHETIC fallback, committed sample."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SimConfig
from .simulate import SensorStream

CMAPSS_URLS = (
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
    "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip",
)
SAMPLE_CONFIG = SimConfig(n_machines=16, days=3.0, seed=20261024, regime_change_day=2.0, weibull_scale_days=5.0)
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"


def probe_cmapss(timeout: float = 10.0) -> dict:
    """Try to reach the public C-MAPSS archive. Only the first bytes are requested."""
    notes = []
    for url in CMAPSS_URLS:
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-1023", "User-Agent": "day14-probe"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                r.read(1024)
            return {"reachable": True, "url": url, "note": f"reachable ({r.status})"}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:  # noqa: PERF203
            notes.append(f"{url}: {type(e).__name__}: {e}")
    return {"reachable": False, "url": None, "note": " | ".join(notes)}


CMAPSS_COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load_cmapss_as_stream(path: str | Path, cycle_s: float = 300.0) -> SensorStream:
    """Map a C-MAPSS ``train_FD00x.txt`` file onto the stream schema (in event-time order, no delays).

    unit -> machine, cycle -> event time (one cycle = ``cycle_s``), s4 (LPT outlet temperature) -> temperature,
    s9 (core speed) -> vibration proxy, s11 (HPC static pressure) -> current proxy. Each unit runs to failure
    at its last cycle. Not used for the reported results (see README).
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLS)
    machine = df["unit"].to_numpy() - 1
    t = (df["cycle"].to_numpy() - 1) * cycle_s
    parts = [(machine, np.full(len(df), s, np.int8), t + s * 0.5, df[c].to_numpy(np.float32)) for s, c in enumerate(("s9", "s4", "s11"))]
    m = np.concatenate([p[0] for p in parts]).astype(np.int32)
    s = np.concatenate([p[1] for p in parts])
    et = np.concatenate([p[2] for p in parts])
    v = np.concatenate([p[3] for p in parts])
    order = np.argsort(et, kind="stable")
    last = df.groupby("unit")["cycle"].max()
    failures = pd.DataFrame(
        {
            "machine": last.index.to_numpy() - 1,
            "lifecycle_start": 0.0,
            "onset_time": np.nan,
            "failure_time": last.to_numpy() * cycle_s,
            "repair_end": last.to_numpy() * cycle_s,
            "sudden": False,
        }
    )
    return SensorStream(
        arrival=et[order],
        event_time=et[order],
        machine=m[order],
        sensor=s[order],
        value=v[order],
        failures=failures,
        horizon_s=float(et.max() + cycle_s),
        n_machines=int(machine.max() + 1),
        meta={"generator": "PUBLIC NASA C-MAPSS replay", "n_events": int(len(m)), "n_failures": int(len(failures))},
    )


def save_sample(stream: SensorStream, directory: Path = SAMPLE_DIR) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    ev = stream.to_frame()
    ev["arrival"] = ev["arrival"].round(2)
    ev["event_time"] = ev["event_time"].round(2)
    ev["value"] = ev["value"].round(3)
    ev.to_csv(directory / "events.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 9, "mtime": 0})
    stream.failures.to_csv(directory / "failures.csv", index=False)
    return {p.name: p.stat().st_size for p in directory.iterdir()}


def load_sample(directory: Path = SAMPLE_DIR) -> SensorStream:
    ev = pd.read_csv(directory / "events.csv.gz")
    fl = pd.read_csv(directory / "failures.csv")
    fl["sudden"] = fl["sudden"].astype(bool)
    cfg = SAMPLE_CONFIG
    return SensorStream(
        arrival=ev["arrival"].to_numpy(np.float64),
        event_time=ev["event_time"].to_numpy(np.float64),
        machine=ev["machine"].to_numpy(np.int32),
        sensor=ev["sensor"].to_numpy(np.int8),
        value=ev["value"].to_numpy(np.float32),
        failures=fl,
        horizon_s=cfg.days * 86400.0,
        n_machines=cfg.n_machines,
        meta={"generator": "SYNTHETIC sample (data/sample)", "seed": cfg.seed, "n_events": int(len(ev)), "n_failures": int(len(fl))},
    )
