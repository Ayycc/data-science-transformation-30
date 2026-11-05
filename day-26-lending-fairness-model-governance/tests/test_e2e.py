import json

from credit26.pipeline import run, small_config


def test_small_end_to_end(tmp_path):
    m = run(small_config(), tmp_path / "res", tmp_path / "data", sample_dir=None, log=lambda s: None)
    saved = json.loads((tmp_path / "res" / "metrics.json").read_text())
    assert saved["recommended"]["model"] == m["recommended"]["model"]
    E = m["explanations"]
    assert E["declined_with_4_reason_codes"] == E["declined_test_applicants"] > 0
    rec = m["models"][m["recommended"]["model"]]
    assert not rec["uses_protected_attribute_at_decision"]
    assert "air_min" in m["models"]["gbm_monotone"]["fairness"] and "air_min" in rec["fairness"]
    mdd = (tmp_path / "res" / "model_development_document.md").read_text()
    for section in [
        "## 2. Data",
        "## 3. Methodology",
        "## 4. Performance testing",
        "## 6. Fair-lending testing",
        "## 8. Limitations",
        "## 9. Ongoing monitoring",
        "## 10. Validation evidence",
    ]:
        assert section in mdd
    for f in ["model_card.md", "report.md", "adverse_action_examples.md", "reason_codes_declined_test.csv", "figures/air_before_after.png"]:
        assert (tmp_path / "res" / f).exists()
