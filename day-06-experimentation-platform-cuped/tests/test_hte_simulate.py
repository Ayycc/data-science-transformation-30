import numpy as np
import pandas as pd
import pytest

from expengine.hte import interaction_test, segment_effects
from expengine.simulate import (
    METRICS,
    EffectSpec,
    ExperimentSpec,
    default_portfolio,
    draw_outcomes,
    draw_users,
    generate_experiment,
    true_effects,
)


def test_zero_effect_truth_is_zero(rng):
    users = draw_users(5000, rng)
    te = true_effects(users, EffectSpec())
    assert all(abs(te[m]["ate"]) < 1e-12 for m in METRICS)


def test_truth_matches_simulated_means():
    users = draw_users(200_000, np.random.default_rng(5))
    eff = EffectSpec(lift_sessions=0.1, lift_purchase=0.2)
    te = true_effects(users, eff)
    rng = np.random.default_rng(6)
    y0 = draw_outcomes(users, np.zeros(200_000, bool), eff, rng)
    y1 = draw_outcomes(users, np.ones(200_000, bool), eff, rng)
    assert y1["sessions"].mean() / y0["sessions"].mean() - 1 == pytest.approx(te["sessions"]["rel"], abs=0.01)
    assert te["sessions"]["rel"] == pytest.approx(0.1, abs=1e-9)
    assert y1["converted"].mean() - y0["converted"].mean() == pytest.approx(te["converted"]["ate"], abs=0.004)


def test_novelty_decays():
    users = draw_users(1000, np.random.default_rng(1))
    eff = EffectSpec(novelty_sessions=0.3, novelty_decay_days=2.0)
    te = true_effects(users, eff)
    assert 0 < te["sessions"]["rel"] < 0.3


def test_srm_bug_drops_treatment_users():
    spec = ExperimentSpec("X", "x", "h", "converted", 20_000, 7, srm_drop_rate=0.5, seed=3)
    df, curve, truth = generate_experiment(spec)
    assert truth["users_dropped_by_bug"] > 1000
    assert (df.variant == "treatment").sum() < (df.variant == "control").sum()
    assert len(curve) == 28


def test_portfolio_scaled():
    p = default_portfolio(0.01)
    assert len(p) == 6 and all(s.n_users >= 400 for s in p)
    assert p[0].to_dict()["exp_id"] == "E01"


def test_sample_is_reproducible(sample_df, tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from generate_data import build_sample

    path = build_sample(tmp_path)
    regen = pd.read_csv(path)
    pd.testing.assert_frame_equal(regen, sample_df)


def test_segment_effects_find_new_user_effect(sample_df):
    df = sample_df.assign(treat=sample_df.variant.eq("treatment"))
    out = segment_effects(df, {"sessions": "pre_sessions"}, ["tenure", "platform"], "treat")
    assert set(out.columns) >= {"p_bh", "significant_bh"}
    new = out[(out.segment == "tenure") & (out.level == "new")].iloc[0]
    old = out[(out.segment == "tenure") & (out.level == "existing")].iloc[0]
    assert new.rel_lift > old.rel_lift
    assert (out.p_bh >= out.p_value - 1e-12).all()
    assert interaction_test(new["diff"], new.se, old["diff"], old.se) < 0.2


def test_segment_effects_empty_when_too_small(sample_df):
    df = sample_df.assign(treat=sample_df.variant.eq("treatment"))
    assert segment_effects(df, {"sessions": "pre_sessions"}, ["tenure"], "treat", min_per_arm=10**7).empty
