from __future__ import annotations

import numpy as np

from iot_stream.config import SimConfig, StreamConfig
from iot_stream.simulate import simulate
from iot_stream.stream import expected_assignment


def test_deterministic_given_seed():
    a = simulate(SimConfig(n_machines=5, days=1.0, seed=3))
    b = simulate(SimConfig(n_machines=5, days=1.0, seed=3))
    assert np.array_equal(a.value, b.value) and np.array_equal(a.arrival, b.arrival)


def test_arrival_order_and_disorder(tiny_stream):
    s = tiny_stream
    assert np.all(np.diff(s.arrival) >= 0)
    assert np.all(s.arrival >= s.event_time)
    lateness, _ = expected_assignment(s, StreamConfig())
    assert (lateness > 60).mean() > 0.01  # genuinely out of order
    assert (lateness > 600).sum() > 0  # some events beyond the allowed lateness


def test_degradation_precedes_failure(tiny_stream):
    s = tiny_stream
    f = s.failures[~s.failures.sudden].iloc[0]
    m = int(f.machine)
    sel = (s.machine == m) & (s.sensor == 0)
    et, v = s.event_time[sel], s.value[sel]
    early = v[(et >= f.lifecycle_start) & (et < f.lifecycle_start + 3 * 3600)].mean()
    late = v[(et > f.failure_time - 1800) & (et < f.failure_time)].mean()
    assert late > 1.5 * early
    # no data while the machine is down for repair
    assert not np.any((et > f.failure_time + 400) & (et < f.repair_end - 400))


def test_sensor_dropouts_present(tiny_stream):
    s = tiny_stream
    n_ticks = int(2 * 86400 / 300)
    assert len(s) < 10 * 3 * n_ticks
