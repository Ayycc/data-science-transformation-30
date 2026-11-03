import json

from recsys24.pipeline import run, small_config


def test_small_end_to_end(tmp_path):
    cfg = small_config()
    res = run(cfg, tmp_path / "results", tmp_path / "data", make_figures=True, sample_dir=tmp_path / "sample", log=lambda m: None)
    assert res["data"]["path_used"] == "synthetic_generator"
    test = res["offline"]["test"]
    assert {"Popularity", "Item-kNN", "ALS", "Two-tower (exact)", "Two-tower (HNSW)", "Two-tower + re-ranker"} <= set(test)
    for m in test.values():
        assert 0 <= m["ndcg@10"] <= 1 and 0 <= m["coverage@10"] <= 1
    assert res["offline"]["latency"]["hnsw"]["end_to_end"]["p99_ms"] > 0
    assert set(res["ope"]["closest_estimator_by_rmse"]) == set(res["ope"]["estimators"])
    saved = json.loads((tmp_path / "results" / "metrics.json").read_text())
    assert saved["offline"]["dataset"]["n_users"] == cfg.world.n_users
    figs = {p.name for p in (tmp_path / "results" / "figures").iterdir()}
    assert {"offline_metrics.png", "ope_estimates_vs_truth.png", "retrieval_latency.png"} <= figs
    assert (tmp_path / "results" / "report.md").read_text().startswith("# Day 24")
    assert all(p.stat().st_size < 1_000_000 for p in (tmp_path / "sample").iterdir())
