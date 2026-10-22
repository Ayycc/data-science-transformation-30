import json

from policy_causal.config import quick_config
from policy_causal.lalonde import load_lalonde, run_lalonde
from policy_causal.pipeline import run


def test_lalonde_from_committed_sample():
    frames, path = load_lalonde("sample")
    assert path in ("cache", "sample")
    r = run_lalonde(frames, seed=0, n_boot=10)
    assert r["n_treated"] == 185
    assert abs(r["experimental_benchmark"]["estimate"] - 1794.34) < 1.0
    assert r["naive_psid"]["estimate"] < 0


def test_pipeline_smoke(tmp_path):
    m = run(quick_config(), results_dir=tmp_path, figures_dir=tmp_path / "fig", lalonde_source="sample")
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["acceptance"]["refutation_tests_reported"] >= 3
    assert (tmp_path / "policy_memo.md").exists() and (tmp_path / "fig" / "event_study.png").exists()
    assert m["acceptance"]["naive_bias_earnings"] < 0
