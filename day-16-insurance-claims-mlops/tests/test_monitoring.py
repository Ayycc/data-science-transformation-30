from __future__ import annotations

from claims_mlops import config as C
from claims_mlops import data as D
from claims_mlops import monitoring as Mon


def test_drift_report_flags_injected_shift_only(small_book, glm, tmp_path):
    ref = D.split_months(small_book, [11, 12])
    ref[Mon.PRED] = glm.predict(ref)
    clean = D.split_months(small_book, [16])
    clean[Mon.PRED] = glm.predict(clean)
    shifted = D.split_months(small_book, [20])
    shifted[Mon.PRED] = glm.predict(shifted)

    r_clean = Mon.drift_report(ref, clean, 16)
    r_shift = Mon.drift_report(ref, shifted, 20, html_path=tmp_path / "r.html")
    assert r_clean.drifted_columns == []
    assert {"DrivAge", "Density"} <= set(r_shift.drifted_columns)
    assert not {"VehPower", "VehBrand", "VehGas", "Region"} & set(r_shift.drifted_columns)
    assert r_shift.drift_share >= C.TriggerPolicy().drift_share
    assert (tmp_path / "r.html").stat().st_size > 10_000


def _drift(share):
    return Mon.DriftResult(1, share, ["DrivAge"] if share else [], {}, {})


def test_trigger_policy():
    pol = C.TriggerPolicy()
    ok = {"ae_ratio": 1.02, "tweedie_deviance": 100.0}
    assert not Mon.retrain_trigger(1, _drift(0.0), ok, 100.0, 5, pol).fire
    assert Mon.retrain_trigger(1, _drift(0.33), ok, 100.0, 5, pol).fire
    assert Mon.retrain_trigger(1, _drift(0.0), {"ae_ratio": 1.15, "tweedie_deviance": 100.0}, 100.0, 5, pol).fire
    assert Mon.retrain_trigger(1, _drift(0.0), {"ae_ratio": 1.0, "tweedie_deviance": 120.0}, 100.0, 5, pol).fire
    cooling = Mon.retrain_trigger(1, _drift(0.33), ok, 100.0, 0, pol)
    assert not cooling.fire and cooling.suppressed_by_cooldown
