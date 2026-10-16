"""End-to-end smoke test on the committed SYNTHETIC sample (data/sample)."""

from __future__ import annotations

import gzip
import json
import re

import pandas as pd

from civic311.config import SAMPLE, SAMPLE_DIR
from civic311.generate import generate


def test_sample_is_small_and_regenerates_exactly():
    files = list(SAMPLE_DIR.iterdir())
    assert sum(f.stat().st_size for f in files) <= 1_000_000
    raw, truth, _ = generate(SAMPLE)
    with gzip.open(SAMPLE_DIR / "requests_raw.csv.gz", "rt") as fh:
        committed = pd.read_csv(fh, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(committed, raw.fillna("").astype(str), check_dtype=False)
    assert len(truth) == len(raw)


def test_every_defect_class_caught_and_traceable(sample_run):
    m, _ = sample_run
    d = m["defect_detection"]
    assert d["classes_present"] == d["defect_classes"] == 15
    assert d["classes_fully_caught"] == 15
    assert d["clean_rows_quarantined"] == 0
    t = m["traceability"]
    assert t["reasons_with_valid_rule_id"] == t["quarantine_reason_rows"]
    assert t["reasons_rederived_by_independent_check"] == t["quarantine_reason_rows"]
    assert t["independent_violations_missing_from_quarantine"] == 0
    assert t["source_lines_found_in_raw"] == t["quarantine_reason_rows"]


def test_great_expectations_certifies_clean_table(sample_run):
    m, _ = sample_run
    g = m["great_expectations"]
    assert not g["raw_success"] and g["clean_success"]
    assert g["pandera_ge_agreement"]["matching"] == g["pandera_ge_agreement"]["comparable_rules"]


def test_incidents_detected_without_false_alarms(sample_run):
    q = sample_run[0]["kpi_anomalies"]["quality"]
    assert q["incidents_detected"] == q["incidents"] == 3
    assert q["false_alarm_days"] == 0


def test_harmonisation_is_exact_with_review(sample_run):
    acc = sample_run[0]["harmonisation"]["accuracy"]
    assert acc["all_real_rows"]["final"]["wrong"] == 0
    assert acc["all_real_rows"]["final"]["unresolved"] == 0
    assert acc["junk_rows_mapped_to_a_category"] == 0


def test_outputs_exist_and_map_is_offline(sample_run):
    m, out = sample_run
    for f in (
        "metrics.json",
        "validation_report.md",
        "validation_report.html",
        "hotspot_map.html",
        "hotspot_map.png",
        "quality_kpis_daily.png",
        "defect_detection.csv",
        "hotspots_h3.geojson",
        "quarantine_examples.csv",
    ):
        assert (out / f).exists(), f
    html = (out / "hotspot_map.html").read_text()
    assert "<polygon" in html and "<script" not in html
    assert not re.search(r"(src|href)\s*=\s*[\"']?(https?:)?//", html)
    assert json.loads((out / "metrics.json").read_text())["data"]["label"] == "SYNTHETIC"
