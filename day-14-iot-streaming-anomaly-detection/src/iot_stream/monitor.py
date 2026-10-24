"""Online learning on windowed features: Half-Space Trees, online RUL regression, drift resets.

Variants
--------
* ``adaptive``  per-window online learning; ADWIN on fleet-median residuals recalibrates the
  per-machine baselines and resets the Half-Space Trees; Page-Hinkley on the RUL model's
  absolute error resets the RUL model.
* ``no_reset``  identical online learning, detectors still run and are logged, but no reset.
* ``static``    "batch-like" reference: models stop learning at ``static_freeze_day`` and
  never reset (stands in for a model retrained on a fixed schedule).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from river import anomaly, drift, preprocessing, tree

from .config import DAY, HOUR, ModelConfig

HST_FEATURES = ("vib_rel", "vib_2h_rel", "vib_std_rel", "vib_slope", "cur_rel", "temp_d", "temp_slope")
DRIFT_CHANNELS = ("vib_rel", "cur_rel", "temp_d")
VARIANTS = ("adaptive", "no_reset", "static")
EWM_ALPHA = 0.25


@dataclass
class MachineState:
    life_start: float
    base_n: int = 0
    base_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # vib, temp, cur, vib_std
    base: list[float] | None = None
    last: dict = field(default_factory=dict)
    hst_run: int = 0
    rul_run: int = 0
    last_warn: float = -math.inf
    ewm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # smoothed vib_rel, temp_d, cur_rel
    pending: deque = field(default_factory=deque)  # (window_end, x_rul, prediction) awaiting labels


def _val(rec: dict, key: str, fallback_key: str, last: dict):
    v = rec.get(key)
    if v is None or v != v:  # NaN -> sensor dropout in this pane
        v = rec.get(fallback_key)
        if v is None or v != v:
            v = last.get(key)
    return v


class FleetMonitor:
    def __init__(self, cfg: ModelConfig, variant: str = "adaptive", slide_s: float = 1800.0):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.cfg = cfg
        self.variant = variant
        self.slide = slide_s
        self.machines: dict[int, MachineState] = {}
        self.failed_at: dict[int, float] = {}
        self.hst_resets = 0
        self.rul_resets = 0
        self._new_hst()
        self._new_rul()
        self._new_adwins()
        self._new_ph()
        self.rows: list[tuple] = []  # machine, window_end, life_start, hst_score, rul_pred, hst_run, rul_run, warn
        self.warnings: list[tuple] = []  # machine, time, reason
        self.drift_events: list[tuple] = []  # time, detector, channel, action
        self.fleet_medians: list[tuple] = []  # window_end, vib_rel, cur_rel, temp_d (before any recalibration)
        self.labels_learned = 0
        self.records_seen = 0
        self.records_skipped = 0

    # ----------------------------------------------------------------- model factories
    def _new_hst(self) -> None:
        c = self.cfg
        self.hst = anomaly.HalfSpaceTrees(
            n_trees=c.hst_trees, height=c.hst_height, window_size=c.hst_window, limits=dict(c.hst_limits), seed=c.seed + self.hst_resets
        )
        self.hst_seen = 0

    def _new_rul(self) -> None:
        self.rul = preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor(grace_period=200, leaf_prediction=self.cfg.rul_leaf)
        self.rul_labels = 0
        self.rul_cap_labels = 0

    def _new_adwins(self) -> None:
        self.adwins = {ch: drift.ADWIN(delta=self.cfg.adwin_delta, clock=32) for ch in DRIFT_CHANNELS}

    def _new_ph(self) -> None:
        c = self.cfg
        self.ph = drift.PageHinkley(min_instances=c.ph_min_instances, delta=c.ph_delta, threshold=c.ph_threshold, mode="up")

    def _learning(self, t: float) -> bool:
        return self.variant != "static" or t < self.cfg.static_freeze_day * DAY

    # ----------------------------------------------------------------- maintenance events
    def on_failure(self, machine: int, t: float) -> None:
        """CMMS failure event: release exact RUL labels for this lifecycle, then reset the machine."""
        st = self.machines.pop(machine, None)
        self.failed_at[machine] = t
        if st is None:
            return
        cap = self.cfg.rul_cap_h
        for tw, x, pred in st.pending:
            self._learn_rul(x, min(cap, (t - tw) / HOUR), pred, tw)

    def _learn_rul(self, x: dict, y: float, pred: float | None, t: float) -> None:
        if not self._learning(t):
            return
        self.rul.learn_one(x, y)
        self.rul_labels += 1
        self.rul_cap_labels += y >= self.cfg.rul_cap_h
        self.labels_learned += 1
        if pred is not None:
            self.ph.update(abs(pred - y))
            if self.ph.drift_detected:
                act = self.variant == "adaptive"
                self.drift_events.append((t, "PageHinkley", "rul_abs_error", "reset_rul_model" if act else "logged_only"))
                if act:
                    self.rul_resets += 1
                    self._new_rul()
                self._new_ph()

    # ----------------------------------------------------------------- features
    def _features(self, st: MachineState, rec: dict) -> dict | None:
        vib = _val(rec, "vib_mean", "vib_2h_mean", st.last)
        temp = _val(rec, "temp_mean", "temp_2h_mean", st.last)
        cur = _val(rec, "cur_mean", "cur_2h_mean", st.last)
        vstd = _val(rec, "vib_std", "vib_2h_std", st.last)
        if vib is None or temp is None or cur is None or vstd is None:
            return None
        st.last.update(vib_mean=vib, temp_mean=temp, cur_mean=cur, vib_std=vstd)
        if st.base is None:
            b = st.base_sum
            b[0] += vib
            b[1] += temp
            b[2] += cur
            b[3] += vstd
            st.base_n += 1
            if st.base_n >= self.cfg.baseline_windows:
                st.base = [v / st.base_n for v in b]
            return None
        bv, bt, bc, bs = st.base
        v2 = rec["vib_2h_mean"] if rec["vib_2h_mean"] == rec["vib_2h_mean"] else vib
        return {
            "vib_rel": vib / bv - 1.0,
            "vib_2h_rel": v2 / bv - 1.0,
            "vib_std_rel": vstd / max(bs, 1e-6) - 1.0,
            "vib_slope": rec["vib_2h_slope"] / bv,
            "cur_rel": cur / bc - 1.0,
            "temp_d": (temp - bt) / 10.0,
            "temp_slope": rec["temp_2h_slope"] / 10.0,
        }

    # ----------------------------------------------------------------- per window
    def process_window(self, end: float, records: list[dict]) -> None:
        cfg = self.cfg
        learning = self._learning(end)
        feats: list[tuple[int, MachineState, dict]] = []
        for rec in records:
            m = rec["machine"]
            self.records_seen += 1
            if end - self.slide < self.failed_at.get(m, -math.inf):
                self.records_skipped += 1  # partial window that straddles a known failure
                continue
            st = self.machines.get(m)
            if st is None:
                st = self.machines[m] = MachineState(life_start=end - self.slide)
            x = self._features(st, rec)
            if x is not None:
                feats.append((m, st, x))

        # fleet-level drift detection on median residuals
        if len(feats) >= 20:
            # ADWIN sees each machine's residual (hundreds of points per window, so an abrupt fleet-wide
            # shift is caught within one window); the fleet median is the recalibration offset.
            medians = {}
            for ch in DRIFT_CHANNELS:
                vals = sorted(f[2][ch] for f in feats)
                medians[ch] = vals[len(vals) // 2]
            self.fleet_medians.append((end, medians["vib_rel"], medians["cur_rel"], medians["temp_d"]))
            fired = [ch for ch in DRIFT_CHANNELS if self._adwin_update(ch, [f[2][ch] for f in feats])]
            if fired:
                act = self.variant == "adaptive"
                for ch in fired:
                    self.drift_events.append((end, "ADWIN", ch, "recalibrate_baselines+reset_hst" if act else "logged_only"))
                if act:
                    self._recalibrate(medians)
                    feats = [(m, st, self._shift(x, medians)) for m, st, x in feats]
                    self.hst_resets += 1
                    self._new_hst()
                    self._new_adwins()

        # label-delay guard: "survived the cap" labels only exist rul_cap_h after a prediction, so early on
        # the model has seen almost only failure labels and under-predicts RUL for everyone.
        rul_ready = self.rul_labels >= cfg.rul_min_labels and self.rul_cap_labels >= cfg.rul_min_cap_labels
        cap_s = cfg.rul_cap_h * HOUR
        for m, st, x in feats:
            hst_ready = self.hst_seen >= cfg.hst_window
            score = self.hst.score_one(x) if hst_ready else math.nan
            if learning:
                self.hst.learn_one(x)
                self.hst_seen += 1
            e = st.ewm
            a = EWM_ALPHA
            e[0] += a * (x["vib_rel"] - e[0])
            e[1] += a * (x["temp_d"] - e[1])
            e[2] += a * (x["cur_rel"] - e[2])
            xr = dict(x)
            xr["vib_ewm"], xr["temp_ewm"], xr["cur_ewm"] = e
            pred = min(max(self.rul.predict_one(xr), 0.0), cfg.rul_cap_h) if rul_ready else None
            st.pending.append((end, xr, pred))
            while st.pending and end - st.pending[0][0] >= cap_s:
                tw, xo, po = st.pending.popleft()
                self._learn_rul(xo, cfg.rul_cap_h, po, tw)
            st.hst_run = st.hst_run + 1 if hst_ready and score >= cfg.hst_threshold else 0
            st.rul_run = st.rul_run + 1 if pred is not None and pred <= cfg.rul_warn_h else 0
            warn = ""
            hst_hit, rul_hit = st.hst_run >= cfg.hst_persist, st.rul_run >= cfg.rul_persist
            if (hst_hit or rul_hit) and end - st.last_warn >= cfg.debounce_h * HOUR:
                warn = "both" if hst_hit and rul_hit else ("hst" if hst_hit else "rul")
                st.last_warn = end
                self.warnings.append((m, end, warn))
            self.rows.append((m, end, st.life_start, score, math.nan if pred is None else pred, st.hst_run, st.rul_run, warn))

    def _adwin_update(self, ch: str, values: list[float]) -> bool:
        """Feed every machine's (clipped) residual for this window; True if ADWIN cut its window."""
        det = self.adwins[ch]
        hit = False
        for v in values:
            det.update(min(max(v, -0.5), 0.5))
            hit = hit or det.drift_detected
        return hit

    def _recalibrate(self, medians: dict) -> None:
        for st in self.machines.values():
            if st.base is not None:
                st.base[0] *= 1.0 + medians["vib_rel"]
                st.base[2] *= 1.0 + medians["cur_rel"]
                st.base[1] += 10.0 * medians["temp_d"]
                st.ewm[0] = (1 + st.ewm[0]) / (1 + medians["vib_rel"]) - 1
                st.ewm[1] -= medians["temp_d"]
                st.ewm[2] = (1 + st.ewm[2]) / (1 + medians["cur_rel"]) - 1

    @staticmethod
    def _shift(x: dict, med: dict) -> dict:
        y = dict(x)
        y["vib_rel"] = (1 + x["vib_rel"]) / (1 + med["vib_rel"]) - 1
        y["vib_2h_rel"] = (1 + x["vib_2h_rel"]) / (1 + med["vib_rel"]) - 1
        y["cur_rel"] = (1 + x["cur_rel"]) / (1 + med["cur_rel"]) - 1
        y["temp_d"] = x["temp_d"] - med["temp_d"]
        return y

    def summary(self) -> dict:
        return {
            "variant": self.variant,
            "window_records_seen": self.records_seen,
            "window_records_skipped_straddling_failure": self.records_skipped,
            "scored_rows": len(self.rows),
            "warnings": len(self.warnings),
            "rul_labels_learned": self.labels_learned,
            "hst_resets": self.hst_resets,
            "rul_resets": self.rul_resets,
            "drift_detections": len(self.drift_events),
        }
