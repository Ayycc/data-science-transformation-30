import json

from privacy25.pipeline import run, small_config


def test_small_end_to_end(tmp_path):
    cfg = small_config()
    cfg.data_mode = "synthetic"
    res = run(cfg, tmp_path / "results", tmp_path / "data", log=lambda m: None)
    m = json.loads((tmp_path / "results" / "metrics.json").read_text())
    assert m["budget"]["within_budget"] and m["budget"]["over_budget_request_refused"]
    assert m["budget"]["epsilon_spent"] <= m["budget"]["epsilon_total"]
    assert len(res["releases"]) == 5
    for r in res["releases"]:
        assert {"tstr", "fidelity", "anonymeter", "mia_dcr", "rare_subgroup"} <= set(r)
    for f in ("privacy_utility_frontier.png", "accuracy_vs_epsilon.png", "dp_model_vs_epsilon.png", "rare_subgroup.png"):
        assert (tmp_path / "results" / "figures" / f).exists()
    assert (tmp_path / "results" / "release_memo.md").read_text().startswith("# Release recommendation memo")
