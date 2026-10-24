from __future__ import annotations

import math

import pandas as pd

from iot_stream.config import ModelConfig, StreamConfig
from iot_stream.evaluate import derive_warnings
from iot_stream.monitor import FleetMonitor
from iot_stream.pipeline import QUICK_MODEL, SCORE_COLS
from iot_stream.stream import replay_models, run_end_to_end, run_windowing


def _rec(m, end, vib=2.0, temp=55.0, cur=20.0):
    r = {"machine": m, "window_end": end}
    for nm, v in (("vib", vib), ("temp", temp), ("cur", cur)):
        r.update({f"{nm}_count": 6, f"{nm}_mean": v, f"{nm}_std": 0.1, f"{nm}_min": v, f"{nm}_max": v})
        r.update({f"{nm}_2h_count": 24, f"{nm}_2h_mean": v, f"{nm}_2h_std": 0.1, f"{nm}_2h_slope": 0.0})
    return r


def test_baseline_then_scoring_and_dropout_imputation():
    cfg = ModelConfig(hst_window=20)
    mon = FleetMonitor(cfg, "adaptive")
    for k in range(1, 40):
        recs = [_rec(m, k * 1800.0) for m in range(30)]
        if k == 30:
            recs[0]["vib_mean"] = math.nan  # sensor dropout in this pane -> falls back to 2 h mean
        mon.process_window(k * 1800.0, recs)
    rows = pd.DataFrame(mon.rows, columns=SCORE_COLS)
    assert rows.window_end.min() == (cfg.baseline_windows + 1) * 1800.0  # first windows build the baseline
    assert rows.hst_score.notna().any()


def test_abrupt_fleet_shift_triggers_adwin_recalibration():
    cfg = ModelConfig(hst_window=50)
    mon = FleetMonitor(cfg, "adaptive")
    for k in range(1, 80):
        vib = 2.0 if k < 50 else 2.6  # +30 % for every machine
        mon.process_window(k * 1800.0, [_rec(m, k * 1800.0, vib=vib * (1 + 0.01 * ((m * 7 + k) % 5))) for m in range(100)])
    adwin = [e for e in mon.drift_events if e[1] == "ADWIN"]
    assert adwin and min(e[0] for e in adwin) >= 50 * 1800.0
    assert mon.hst_resets >= 1
    # after recalibration the fleet residual is back near zero
    assert abs(mon.fleet_medians[-1][1]) < 0.05

    ctrl = FleetMonitor(cfg, "no_reset")
    for k in range(1, 80):
        vib = 2.0 if k < 50 else 2.6
        ctrl.process_window(k * 1800.0, [_rec(m, k * 1800.0, vib=vib * (1 + 0.01 * ((m * 7 + k) % 5))) for m in range(100)])
    assert ctrl.hst_resets == 0 and abs(ctrl.fleet_medians[-1][1] - 0.3) < 0.05


def test_failure_releases_exact_rul_labels():
    cfg = ModelConfig(hst_window=10)
    mon = FleetMonitor(cfg, "adaptive")
    for k in range(1, 20):
        mon.process_window(k * 1800.0, [_rec(0, k * 1800.0), _rec(1, k * 1800.0)])
    pending = len(mon.machines[0].pending)
    mon.on_failure(0, 20 * 1800.0)
    assert mon.labels_learned == pending and 0 not in mon.machines
    # a window that straddles the failure is ignored; the next lifecycle starts a fresh baseline
    mon.process_window(20 * 1800.0, [_rec(0, 20 * 1800.0)])
    assert 0 not in mon.machines
    mon.process_window(25 * 1800.0, [_rec(0, 25 * 1800.0)])
    assert mon.machines[0].base is None and mon.machines[0].base_n == 1


def test_online_warnings_equal_rederived_and_e2e_equals_replay(sample_stream):
    scfg = StreamConfig()
    a = FleetMonitor(QUICK_MODEL, "adaptive")
    run_end_to_end(sample_stream, scfg, a)
    b = FleetMonitor(QUICK_MODEL, "adaptive")
    replay_models(run_windowing(sample_stream, scfg).window_records, sample_stream.failures, b)
    assert a.rows == b.rows or pd.DataFrame(a.rows).equals(pd.DataFrame(b.rows))
    rows = pd.DataFrame(a.rows, columns=SCORE_COLS)
    der = derive_warnings(rows, QUICK_MODEL, "combined")
    assert sorted(zip(der.machine, der.t, strict=True)) == sorted((m, t) for m, t, _ in a.warnings)
