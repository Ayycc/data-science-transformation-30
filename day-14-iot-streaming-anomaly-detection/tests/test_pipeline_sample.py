from __future__ import annotations

import json

from iot_stream.pipeline import run


def test_quick_pipeline_end_to_end(tmp_path):
    m = run(quick=True, results_dir=tmp_path / "res", db_path=tmp_path / "m.duckdb", make_plots=True)
    acc = m["acceptance"]
    assert acc["late_events_within_lateness_correctly_assigned"]
    assert m["windowing"]["stream_vs_batch_sql"]["count_mismatches"] == 0
    assert m["windowing"]["stream_vs_batch_sql"]["max_abs_diff"] < 1e-6
    assert m["headline"]["online_rule_matches_rederived"]
    assert m["throughput"]["windowing_events_per_sec"] > 10_000
    saved = json.loads((tmp_path / "res" / "metrics.json").read_text())
    assert {"warning_eval", "exactly_once_demo", "throughput"} <= set(saved)
    assert (tmp_path / "res" / "figures" / "lateness_hist.png").exists()
