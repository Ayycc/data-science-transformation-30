import json

import pandas as pd

from extract_eval.extractors.rules import RulesExtractor
from extract_eval.pipeline import ROOT, load_corpus, main, run_extractor


def test_pipeline_runs_offline_on_small_corpus(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    m = main(n_docs=60, res_dir=tmp_path / "results", data_dir=tmp_path / "data", write_sample=False)
    assert m["llm"]["status"].startswith("skipped")
    assert set(m["extractors"]) == {"regex_strict", "rules_repair"}
    for _name, d in m["downstream"].items():
        assert d["invalid_written_downstream"] == 0
        assert d["accepted"] + d["quarantined"] == 60
    t = pd.read_csv(tmp_path / "results" / "per_field_metrics.csv")
    assert {"exact_acc", "fuzzy_acc", "coverage"} <= set(t.columns) and len(t) == 2 * 23
    saved = json.loads((tmp_path / "results" / "metrics.json").read_text())
    assert 0 <= saved["extractors"]["rules_repair"]["calibration_test"]["ece_calibrated"] <= 1
    assert (tmp_path / "results" / "figures" / "reliability.png").exists()


def test_committed_sample_extracts_well():
    texts, golds = load_corpus(ROOT / "data" / "sample")
    res = run_extractor(RulesExtractor("rules"), texts)
    seen = [
        r for r in res if not golds[r.doc_id]["meta"]["template"].endswith("_unseen") and not golds[r.doc_id]["meta"]["scanned"]
    ]
    assert seen and all(r.valid for r in seen)
