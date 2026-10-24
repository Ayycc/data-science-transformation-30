"""Event-time windows, watermarks and allowed lateness (acceptance criterion: late events within the
allowed lateness are correctly assigned)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from iot_stream.windowing import EventTimeWindower

SLIDE, SIZE, LATE = 60.0, 180.0, 30.0


def feed(w: EventTimeWindower, events: list[tuple[int, int, float, float]]) -> list[dict]:
    out = []
    for i, (m, s, et, v) in enumerate(events):
        w.process_batch([m], [s], [et], [v], offset=i)
        out.extend(w.advance())
    out.extend(w.flush())
    return out


def by_key(records: list[dict]) -> dict[tuple[int, float], dict]:
    return {(r["machine"], r["window_end"]): r for r in records}


def test_late_event_within_lateness_lands_in_its_own_window():
    w = EventTimeWindower(SLIDE, SIZE, LATE)
    events = [(0, 0, 10.0, 1.0), (0, 0, 50.0, 2.0), (0, 0, 70.0, 3.0), (0, 0, 55.0, 10.0), (0, 0, 130.0, 4.0)]
    # 55 arrives after 70: 15 s late (< 30 s allowed) -> must be in [0, 60)
    rec = by_key(feed(w, events))
    assert rec[(0, 60.0)]["vib_count"] == 3
    assert rec[(0, 60.0)]["vib_mean"] == pytest.approx((1 + 2 + 10) / 3)
    assert rec[(0, 120.0)]["vib_count"] == 1
    assert w.n_late_accepted == 1 and w.n_dropped == 0


def test_event_beyond_lateness_after_window_closed_is_dropped():
    w = EventTimeWindower(SLIDE, SIZE, LATE)
    # max event time 100 -> watermark 70 >= 60: window [0, 60) closed; event at 20 is 80 s late
    events = [(0, 0, 10.0, 1.0), (0, 0, 100.0, 2.0), (0, 0, 20.0, 99.0)]
    rec = by_key(feed(w, events))
    assert rec[(0, 60.0)]["vib_count"] == 1
    assert w.n_dropped == 1 and w.dead_letter == [2]


def test_beyond_lateness_but_window_still_open_is_kept():
    w = EventTimeWindower(SLIDE, SIZE, LATE)
    # 50 s behind max (85) but its window [0, 60) has end 60 > watermark 55, so it is still open
    rec = by_key(feed(w, [(0, 0, 85.0, 1.0), (0, 0, 35.0, 5.0)]))
    assert rec[(0, 60.0)]["vib_count"] == 1
    assert w.n_dropped == 0


def test_window_fires_only_when_watermark_passes_end():
    w = EventTimeWindower(SLIDE, SIZE, LATE)
    w.process_batch([0], [0], [10.0], [1.0])
    w.process_batch([0], [0], [89.0], [1.0])  # watermark 59 < 60
    assert w.advance() == []
    w.process_batch([0], [0], [90.0], [1.0])  # watermark 60 -> [0, 60) fires
    fired = w.advance()
    assert [r["window_end"] for r in fired] == [60.0]


def _random_stream(n: int, seed: int):
    rng = random.Random(seed)
    ev = []
    for i in range(n):
        et = i * 1.7 + rng.random()
        delay = rng.expovariate(1 / 3.0) if rng.random() < 0.9 else rng.uniform(0, 90)
        ev.append((et + delay, rng.randrange(4), rng.randrange(3), et, rng.gauss(10, 2)))
    ev.sort()
    return [(m, s, et, v) for _, m, s, et, v in ev]


def _oracle(events, slide, size, lateness):
    """Brute force: replay the watermark rule event by event, then aggregate by definition."""
    max_et = -math.inf
    kept = []
    for m, s, et, v in events:
        max_et = max(max_et, et)
        if (math.floor(et / slide) + 1) * slide > max_et - lateness:
            kept.append((m, s, et, v))
    return kept


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_bruteforce_tumbling_and_sliding(seed):
    events = _random_stream(3000, seed)
    w = EventTimeWindower(SLIDE, SIZE, LATE)
    recs = by_key(feed(w, events))
    kept = _oracle(events, SLIDE, SIZE, LATE)
    assert w.n_dropped == len(events) - len(kept)
    names = ("vib", "temp", "cur")
    for (m, end), r in recs.items():
        for s, nm in enumerate(names):
            tum = [v for mm, ss, et, v in kept if mm == m and ss == s and end - SLIDE <= et < end]
            sl = [v for mm, ss, et, v in kept if mm == m and ss == s and end - SIZE <= et < end]
            assert r[f"{nm}_count"] == len(tum)
            assert r[f"{nm}_2h_count"] == len(sl)
            if tum:
                assert r[f"{nm}_mean"] == pytest.approx(np.mean(tum))
                assert r[f"{nm}_max"] == pytest.approx(max(tum))
                assert r[f"{nm}_std"] == pytest.approx(np.std(tum), abs=1e-6)
            if sl:
                assert r[f"{nm}_2h_mean"] == pytest.approx(np.mean(sl))
    # every (machine, pane) with kept data produced exactly one record
    expected_keys = {(m, (math.floor(et / SLIDE) + 1) * SLIDE) for m, s, et, v in kept}
    assert set(recs) == expected_keys


def test_result_independent_of_batch_size():
    events = _random_stream(2000, 9)
    ref = None
    for bs in (1, 7, 64, 2000):
        w = EventTimeWindower(SLIDE, SIZE, LATE)
        out = []
        for i in range(0, len(events), bs):
            chunk = events[i : i + bs]
            w.process_batch(*map(list, zip(*chunk, strict=True)), offset=i)
            out.extend(w.advance())
        out.extend(w.flush())
        got = sorted(
            (r["machine"], r["window_end"], r["vib_count"], round(r["vib_2h_mean"], 9) if r["vib_2h_count"] else None) for r in out
        )
        ref = ref or got
        assert got == ref


def test_slope_sign_tracks_trend():
    w = EventTimeWindower(SLIDE, SIZE, LATE)
    events = [(0, 0, float(t), 1.0 + t / 60.0) for t in range(0, 240, 5)]
    rec = by_key(feed(w, events))
    assert rec[(0, 240.0)]["vib_2h_slope"] > 0


def test_invalid_window_spec():
    with pytest.raises(ValueError):
        EventTimeWindower(60.0, 150.0, 10.0)
