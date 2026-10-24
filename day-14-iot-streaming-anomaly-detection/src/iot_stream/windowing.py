"""Event-time windowing with watermarks and allowed lateness.

Design (pane-based, as in Flink/Dataflow "panes" or Li et al. 2005 "No pane, no gain"):

* Every event updates exactly one *pane* of length ``slide`` keyed by (pane index, machine).
* A **tumbling** window is one pane. A **sliding** window of length ``size`` (a multiple of
  ``slide``) is the merge of the last ``size / slide`` panes, so sliding aggregates cost O(1)
  per event instead of O(size / slide).
* The **watermark** is ``max event time seen - allowed_lateness``. A window ``[start, end)``
  fires when ``end <= watermark``. An event whose pane end is already ``<= watermark`` (the
  value at the moment it arrives) is too late and goes to the dead-letter list.
* Consequence: an event at most ``allowed_lateness`` behind the newest event is ALWAYS
  assigned to its correct window, regardless of arrival order or batch size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .config import SENSORS

STRIDE = 5  # per sensor accumulator: count, sum, sum of squares, min, max
_PREFIX = ("vib", "temp", "cur")
assert len(_PREFIX) == len(SENSORS)


def _new_acc(n_sensors: int) -> list[float]:
    return [0.0, 0.0, 0.0, math.inf, -math.inf] * n_sensors


class EventTimeWindower:
    """Keyed (machine) tumbling + sliding event-time windows sharing one pane store."""

    def __init__(self, slide_s: float, sliding_size_s: float, allowed_lateness_s: float, n_sensors: int = 3):
        k = sliding_size_s / slide_s
        if abs(k - round(k)) > 1e-9 or k < 1:
            raise ValueError("sliding window size must be a positive multiple of the slide")
        self.slide = float(slide_s)
        self.k = int(round(k))
        self.lateness = float(allowed_lateness_s)
        self.n_sensors = n_sensors
        self.panes: dict[int, dict[int, list[float]]] = {}
        self.max_event_time = -math.inf
        self.next_pane: int | None = None
        self.n_in = 0
        self.n_out_of_order = 0
        self.n_late_accepted = 0
        self.n_dropped = 0
        self.n_windows_fired = 0
        self.dead_letter: list[int] = []  # global offsets of events dropped as too late

    @property
    def watermark(self) -> float:
        return self.max_event_time - self.lateness

    def process_batch(
        self,
        machines: Sequence[int],
        sensors: Sequence[int],
        event_times: Sequence[float],
        values: Sequence[float],
        offset: int = 0,
    ) -> None:
        """Assign a micro-batch of events to panes (does not fire windows)."""
        slide, lateness, panes, ns = self.slide, self.lateness, self.panes, self.n_sensors
        max_et = self.max_event_time
        nxt = self.next_pane
        ooo = late_ok = dropped = 0
        dead = self.dead_letter
        for i, (m, s, et, v) in enumerate(zip(machines, sensors, event_times, values, strict=True)):
            if et > max_et:
                max_et = et
                is_late = False
            else:
                is_late = et < max_et
                ooo += is_late
            p = int(et // slide)
            if (p + 1) * slide <= max_et - lateness:
                dropped += 1
                dead.append(offset + i)
                continue
            late_ok += is_late
            pane = panes.get(p)
            if pane is None:
                pane = panes[p] = {}
                if nxt is None or p < nxt:
                    nxt = p
            acc = pane.get(m)
            if acc is None:
                acc = pane[m] = _new_acc(ns)
            o = s * STRIDE
            acc[o] += 1.0
            acc[o + 1] += v
            acc[o + 2] += v * v
            if v < acc[o + 3]:
                acc[o + 3] = v
            if v > acc[o + 4]:
                acc[o + 4] = v
        self.max_event_time = max_et
        self.next_pane = nxt
        self.n_in += len(machines)
        self.n_out_of_order += ooo
        self.n_late_accepted += late_ok
        self.n_dropped += dropped

    def advance(self) -> list[dict]:
        """Fire every window whose end is <= the current watermark."""
        return self._fire(self.watermark)

    def flush(self) -> list[dict]:
        """End of stream: the watermark jumps to +inf and all remaining windows fire."""
        if not self.panes:
            return []
        return self._fire((max(self.panes) + 1) * self.slide)

    def _fire(self, upto: float) -> list[dict]:
        out: list[dict] = []
        slide, k, panes = self.slide, self.k, self.panes
        while self.next_pane is not None and (self.next_pane + 1) * slide <= upto:
            p = self.next_pane
            tum = panes.get(p)
            if tum:
                window = [panes.get(q) for q in range(p - k + 1, p + 1)]
                end = (p + 1) * slide
                for m, acc in tum.items():
                    out.append(self._record(m, end, acc, [w.get(m) if w else None for w in window]))
            panes.pop(p - k + 1, None)
            self.next_pane = p + 1
            if not panes:
                self.next_pane = None
        self.n_windows_fired += len(out)
        return out

    def _record(self, machine: int, end: float, acc: list[float], window: list[list[float] | None]) -> dict:
        rec: dict = {"machine": machine, "window_end": end}
        hours_per_pane = self.slide / 3600.0
        for s, name in enumerate(_PREFIX):
            o = s * STRIDE
            c = acc[o]
            if c > 0:
                mean = acc[o + 1] / c
                var = max(acc[o + 2] / c - mean * mean, 0.0)
                rec[f"{name}_count"] = int(c)
                rec[f"{name}_mean"] = mean
                rec[f"{name}_std"] = math.sqrt(var)
                rec[f"{name}_min"] = acc[o + 3]
                rec[f"{name}_max"] = acc[o + 4]
            else:
                rec[f"{name}_count"] = 0
                rec[f"{name}_mean"] = rec[f"{name}_std"] = rec[f"{name}_min"] = rec[f"{name}_max"] = math.nan
            # sliding window = merge of panes; slope = least squares over pane means
            cnt = tot = sq = 0.0
            xs: list[float] = []
            ys: list[float] = []
            for j, pa in enumerate(window):
                if pa is not None and pa[o] > 0:
                    cnt += pa[o]
                    tot += pa[o + 1]
                    sq += pa[o + 2]
                    xs.append(j * hours_per_pane)
                    ys.append(pa[o + 1] / pa[o])
            rec[f"{name}_2h_count"] = int(cnt)
            if cnt > 0:
                mean2 = tot / cnt
                rec[f"{name}_2h_mean"] = mean2
                rec[f"{name}_2h_std"] = math.sqrt(max(sq / cnt - mean2 * mean2, 0.0))
            else:
                rec[f"{name}_2h_mean"] = rec[f"{name}_2h_std"] = math.nan
            if len(xs) >= 2:
                mx = sum(xs) / len(xs)
                my = sum(ys) / len(ys)
                den = sum((x - mx) ** 2 for x in xs)
                rec[f"{name}_2h_slope"] = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den
            else:
                rec[f"{name}_2h_slope"] = 0.0
        return rec

    def counters(self) -> dict:
        return {
            "events_in": self.n_in,
            "events_out_of_order": self.n_out_of_order,
            "late_events_assigned": self.n_late_accepted,
            "events_dropped_too_late": self.n_dropped,
            "window_records_fired": self.n_windows_fired,
        }
