import json

import numpy as np

from invopt import data as dm
from invopt.pipeline import prepare, run, sim_inputs
from invopt.simulation import simulate


def test_simulation_conserves_units_and_is_deterministic(small_cfg):
    ds = dm.generate(small_cfg)
    prep = prepare(small_cfg, ds)
    inp = sim_inputs(small_cfg, prep, ds, "optimised")
    a = simulate(inp, allocator="milp", time_limit_s=5)
    b = simulate(inp, allocator="milp", time_limit_s=5)
    assert a.total_cost == b.total_cost
    assert np.all(a.served <= a.demand + 1e-9)
    assert 0 < a.fill_rate <= 1
    assert np.all(a.weekly_spend <= inp.budget + 1e-6)
    h = simulate(sim_inputs(small_cfg, prep, ds, "heuristic"), allocator="greedy")
    assert 0 < h.fill_rate <= 1
    # with huge starting stock nothing is lost
    inp.init_on_hand = inp.init_on_hand + 1e6
    assert simulate(inp, allocator="greedy").fill_rate == 1.0


def test_pipeline_smoke(small_cfg, tmp_path):
    m = run(small_cfg, tmp_path / "results", tmp_path / "data", sensitivity=True, figures=True)
    assert (tmp_path / "results" / "metrics.json").exists()
    saved = json.loads((tmp_path / "results" / "metrics.json").read_text())
    assert saved["label"] == "SYNTHETIC"
    assert set(m["simulation"]) == {"heuristic", "optimised_greedy", "optimised_milp"}
    assert m["milp"]["weekly_solves"] > 0
    assert (tmp_path / "results" / "cost_breakdown.png").exists()
    assert (tmp_path / "results" / "sensitivity_lead_time_cv.csv").exists()
