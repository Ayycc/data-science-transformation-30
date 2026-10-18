from __future__ import annotations

import json

from retention.pipeline import run


def test_pipeline_end_to_end_fast(cfg, tmp_path):
    m = run(cfg, tmp_path / "results", None, n_customers=20_000, fast=True)
    saved = json.loads((tmp_path / "results" / "metrics.json").read_text())
    assert saved["headline"] == json.loads(json.dumps(saved["headline"]))
    for f in [
        "km_by_segment.png",
        "schoenfeld_contract.png",
        "qini_curves.png",
        "policy_comparison.csv",
        "clv_model_vs_true.png",
    ]:
        assert (tmp_path / "results" / f).exists()
    assert m["survival"]["ph_violations_before"] and not m["survival"]["ph_violations_after"]
    pols = m["policy"]["policies"]
    assert pols["oracle (true net value)"]["net_incremental_revenue_true"] >= max(
        v["net_incremental_revenue_true"] for k, v in pols.items() if not k.startswith("oracle") and "no budget" not in k
    )
    for k, v in pols.items():
        if "no budget" not in k:
            assert v["spend_estimated"] <= cfg["economics"]["budget"] + 1e-6
    assert m["clv"]["pearson_r_vs_true"] > 0.9
