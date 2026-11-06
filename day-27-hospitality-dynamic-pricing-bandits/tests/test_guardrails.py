import numpy as np
import pytest

from pricing27.policies import GRID, Guardrails, RandomPolicy
from pricing27.simulator import competitor_price, context_for, make_world
from pricing27.study import POLICY_ORDER, StudyConfig, make_policies, run_policy


def test_candidate_sets_always_feasible_property():
    g = Guardrails()
    rng = np.random.default_rng(0)
    for _ in range(5000):
        static = float(rng.uniform(20, 900))
        prev = None if rng.random() < 0.1 else float(rng.uniform(g.floor, g.ceiling))
        c = g.candidates(static, prev)
        assert len(c) >= 1
        assert all(g.is_ok(p, prev) for p in c)
        assert c.min() >= g.floor - 1e-9 and c.max() <= g.ceiling + 1e-9


def _check_path(prices, g):
    assert prices.min() >= g.floor - 1e-6 and prices.max() <= g.ceiling + 1e-6
    rel = np.abs(np.diff(prices, axis=0)) / prices[:-1]
    assert rel.max() <= g.max_change + 1e-6


@pytest.mark.parametrize("name", POLICY_ORDER)
def test_no_policy_ever_violates_guardrails(name, cfg, models):
    """Includes greedy_naive, which pushes to the ceiling, and cold-start bandits."""
    g = Guardrails()
    scfg = StudyConfig(n_seeds=1, n_nights=90, warm_start=name != "linucb")
    pol = make_policies((name,), models, cfg, scfg)[name]
    w = make_world(365, 90, 11, cfg)
    u = np.random.default_rng(1).random((90, 3))
    r = run_policy(pol, w, cfg, g, models, u, 5)
    _check_path(r["prices"], g)
    assert r["guardrail_violations_posted"] == 0 and r["guardrail_interventions"] == 0


def test_tight_guardrails_and_random_policy_with_extreme_market(cfg, models):
    from dataclasses import replace

    g = Guardrails(floor=120.0, ceiling=180.0, max_change=0.05)
    hot = replace(cfg, event_prob=0.5, comp_noise=0.5)
    w = make_world(365, 120, 4, hot)
    u = np.random.default_rng(2).random((120, 3))
    r = run_policy(RandomPolicy(), w, hot, g, models, u, 3)
    _check_path(r["prices"], g)
    assert r["guardrail_violations_posted"] == 0


def test_safety_layer_clips_a_rogue_policy(cfg, models):
    class Rogue(RandomPolicy):
        name = "rogue"

        def choose(self, ctx, cands, R, v_next, rng):
            return 10_000.0

    g = Guardrails()
    w = make_world(365, 20, 4, cfg)
    r = run_policy(Rogue(), w, cfg, g, models, np.random.default_rng(2).random((20, 3)), 3)
    _check_path(r["prices"], g)
    assert r["guardrail_interventions"] > 0 and r["guardrail_violations_posted"] == 0


def test_grid_and_context(cfg):
    assert np.isclose(GRID[0], 0.6) and np.isclose(GRID[-1], 1.8) and len(GRID) == 13
    w = make_world(0, 3, 0, cfg)
    assert context_for(w, 0, 2, competitor_price(w, 0)).features().shape == (7,)
