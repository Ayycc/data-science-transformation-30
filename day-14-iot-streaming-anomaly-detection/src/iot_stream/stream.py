"""In-process asyncio dataflow: source -> event-time windowing -> online models.

The source reads micro-batches in arrival order (the shape a Kafka consumer ``poll`` returns).
Bounded queues give back-pressure between stages. No broker is needed; ``scripts/kafka_replay.py``
shows the same windower fed from Redpanda/Kafka.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import StreamConfig
from .monitor import FleetMonitor
from .simulate import SensorStream
from .windowing import EventTimeWindower


@dataclass
class RunResult:
    seconds: float
    events: int
    window_records: list[dict] = field(default_factory=list)
    windower: EventTimeWindower | None = None

    @property
    def events_per_sec(self) -> float:
        return self.events / self.seconds if self.seconds > 0 else float("nan")


def make_windower(cfg: StreamConfig) -> EventTimeWindower:
    return EventTimeWindower(cfg.slide_s, cfg.sliding_size_s, cfg.allowed_lateness_s)


async def _source(stream: SensorStream, q: asyncio.Queue, batch_size: int, start: int = 0) -> None:
    n = len(stream)
    for i in range(start, n, batch_size):
        j = min(i + batch_size, n)
        batch = (
            stream.machine[i:j].tolist(),
            stream.sensor[i:j].tolist(),
            stream.event_time[i:j].tolist(),
            stream.value[i:j].tolist(),
        )
        await q.put((i, batch))
    await q.put(None)


async def _window_stage(qin: asyncio.Queue, qout: asyncio.Queue | None, w: EventTimeWindower, collect: list | None) -> None:
    while True:
        item = await qin.get()
        if item is None:
            break
        off, (m, s, et, v) = item
        w.process_batch(m, s, et, v, offset=off)
        fired = w.advance()
        if fired:
            if collect is not None:
                collect.extend(fired)
            if qout is not None:
                await qout.put(fired)
    fired = w.flush()
    if collect is not None:
        collect.extend(fired)
    if qout is not None:
        if fired:
            await qout.put(fired)
        await qout.put(None)


def group_by_end(records: list[dict]) -> dict[float, list[dict]]:
    out: dict[float, list[dict]] = {}
    for r in records:
        out.setdefault(r["window_end"], []).append(r)
    return out


class ModelDriver:
    """Merges window records with CMMS failure events in event-time order and feeds a FleetMonitor."""

    def __init__(self, monitor: FleetMonitor, failures: pd.DataFrame):
        self.monitor = monitor
        f = failures.sort_values("failure_time")
        self.fail = list(zip(f["failure_time"].tolist(), f["machine"].tolist(), strict=True))
        self.fi = 0

    def feed(self, records: list[dict]) -> None:
        for end, recs in sorted(group_by_end(records).items()):
            while self.fi < len(self.fail) and self.fail[self.fi][0] <= end:
                t, m = self.fail[self.fi]
                self.monitor.on_failure(m, t)
                self.fi += 1
            self.monitor.process_window(end, recs)

    def close(self) -> None:
        while self.fi < len(self.fail):
            t, m = self.fail[self.fi]
            self.monitor.on_failure(m, t)
            self.fi += 1


async def _model_stage(qin: asyncio.Queue, driver: ModelDriver) -> None:
    while True:
        item = await qin.get()
        if item is None:
            break
        driver.feed(item)
    driver.close()


def run_windowing(stream: SensorStream, cfg: StreamConfig, collect: bool = True) -> RunResult:
    """Source + windowing only (processor throughput)."""
    w = make_windower(cfg)
    records: list[dict] = []

    async def main() -> None:
        q: asyncio.Queue = asyncio.Queue(maxsize=8)
        await asyncio.gather(_source(stream, q, cfg.batch_size), _window_stage(q, None, w, records if collect else None))

    t0 = time.perf_counter()
    asyncio.run(main())
    return RunResult(time.perf_counter() - t0, len(stream), records, w)


def run_end_to_end(stream: SensorStream, cfg: StreamConfig, monitor: FleetMonitor) -> RunResult:
    """Source + windowing + online models in one dataflow."""
    w = make_windower(cfg)
    driver = ModelDriver(monitor, stream.failures)

    async def main() -> None:
        q1: asyncio.Queue = asyncio.Queue(maxsize=8)
        q2: asyncio.Queue = asyncio.Queue(maxsize=8)
        await asyncio.gather(_source(stream, q1, cfg.batch_size), _window_stage(q1, q2, w, None), _model_stage(q2, driver))

    t0 = time.perf_counter()
    asyncio.run(main())
    return RunResult(time.perf_counter() - t0, len(stream), [], w)


def replay_models(records: list[dict], failures: pd.DataFrame, monitor: FleetMonitor) -> float:
    """Model stage only, on already-windowed records (used for the ablation variants)."""
    driver = ModelDriver(monitor, failures)
    t0 = time.perf_counter()
    driver.feed(records)  # grouped by window end, so fleet medians see every machine
    driver.close()
    return time.perf_counter() - t0


def expected_assignment(stream: SensorStream, cfg: StreamConfig) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised oracle: lateness of every event and whether the watermark rule must accept it."""
    cummax = np.maximum.accumulate(stream.event_time)
    lateness = cummax - stream.event_time
    pane_end = (np.floor(stream.event_time / cfg.slide_s) + 1) * cfg.slide_s
    accept = pane_end > cummax - cfg.allowed_lateness_s
    return lateness, accept
