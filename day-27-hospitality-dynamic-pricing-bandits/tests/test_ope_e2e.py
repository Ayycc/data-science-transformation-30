import json

import numpy as np

from pricing27.pipeline import LOG_SEED, _ope, run, small_config
from pricing27.simulator import make_world


def test_ope_dr_close_to_truth(logs, models, cfg):
    res = _ope(logs, models, cfg)
    assert res["_n_cells"] == int(logs.price_test.sum())
    errs = [abs(v["dr_rel_error"]) for k, v in res.items() if not k.startswith("_")]
    assert max(errs) < 0.10
    assert make_world(0, 365, LOG_SEED, cfg).u.shape == (365,)


def test_small_end_to_end(tmp_path):
    m = run(small_config(), tmp_path / "res", tmp_path / "data", sample_dir=None, log=lambda s: None)
    saved = json.loads((tmp_path / "res" / "metrics.json").read_text())
    assert saved["acceptance"]["guardrail_violations_total"] == 0
    P = saved["study"]["policies"]
    assert set(P) >= {"static", "greedy_iv", "eps_greedy", "linucb", "thompson", "oracle"}
    u = P["thompson"]["uplift_vs_static"]
    assert u["t_ci"][0] <= u["mean"] <= u["t_ci"][1]
    assert np.isclose(P["static"]["uplift_vs_static"]["mean"], 0.0)
    for f in ["report.md", "policy_summary.csv", "figures/regret_curves.png", "figures/uplift_vs_static.png", "figures/regret_curves.html"]:
        assert (tmp_path / "res" / f).exists()
    assert m["data"]["path_used"] == "synthetic_simulator"
