from __future__ import annotations

from dataclasses import replace

import numpy as np

from iot_stream.config import StreamConfig
from iot_stream.data import SAMPLE_DIR, load_cmapss_as_stream
from iot_stream.sink import exactly_once_demo
from iot_stream.stream import run_windowing


def test_exactly_once_vs_at_least_once(sample_stream):
    out = exactly_once_demo(sample_stream, replace(StreamConfig(), batch_size=1024), checkpoint_every=5, crash_after_batch=23)
    assert out["transactional"]["identical_to_clean_run"] and out["transactional"]["duplicate_rows"] == 0
    assert out["at_least_once"]["duplicate_rows"] > 0 and out["at_least_once"]["missing_rows"] == 0


def test_sample_small_and_loadable(sample_stream):
    assert sum(p.stat().st_size for p in SAMPLE_DIR.iterdir()) <= 1_000_000
    assert len(sample_stream) > 10_000 and len(sample_stream.failures) > 0
    assert np.all(np.diff(sample_stream.arrival) >= 0)


def test_cmapss_loader_on_fixture(tmp_path):
    rows = []
    for unit, n in ((1, 40), (2, 30)):
        for c in range(1, n + 1):
            rows.append(" ".join(str(x) for x in [unit, c, 0.0, 0.0, 100.0] + [500.0 + c * 0.1 + i for i in range(21)]))
    p = tmp_path / "train_FD001.txt"
    p.write_text("\n".join(rows) + "\n")
    s = load_cmapss_as_stream(p)
    assert len(s) == 3 * 70 and set(s.failures.machine) == {0, 1}
    recs = run_windowing(s, StreamConfig()).window_records
    assert sum(r["vib_count"] for r in recs) == 70
