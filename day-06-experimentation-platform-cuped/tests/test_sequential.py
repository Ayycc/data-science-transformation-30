import numpy as np
import pytest

from expengine.sequential import (
    always_valid_ci,
    msprt_lambda,
    msprt_pvalues,
    msprt_sequence,
    naive_pvalues,
    tau2_from_effect,
)


def test_lambda_formula():
    lam = msprt_lambda(0.0, 1.0, 1.0)
    assert lam == pytest.approx(np.sqrt(0.5))
    assert msprt_lambda(3.0, 1.0, 1.0) > lam


def test_pvalues_monotone_and_bounded(rng):
    diff = rng.standard_normal((5, 30)) * 0.1
    var = np.full((5, 30), 0.01)
    p = msprt_pvalues(diff, var, 0.01)
    assert (np.diff(p, axis=1) <= 0).all() and (p <= 1).all() and (p > 0).all()


def test_ci_contains_estimate_and_shrinks():
    n = np.arange(1, 51) * 100.0
    var = 1.0 / n
    diff = np.full(50, 0.05)
    lo, hi = always_valid_ci(diff, var, 0.01)
    assert (lo <= 0.05).all() and (hi >= 0.05).all()
    assert (hi - lo)[-1] < (hi - lo)[0]


def test_always_valid_ci_wider_than_fixed():
    lo, hi = always_valid_ci(np.array([0.0]), np.array([0.01]), 0.01)
    assert hi[0] > 1.96 * 0.1


def test_type1_control_under_peeking(rng):
    reps, looks, m = 2000, 30, 50
    y_c = rng.standard_normal((reps, looks * m)).reshape(reps, looks, m)
    y_t = rng.standard_normal((reps, looks * m)).reshape(reps, looks, m)
    n = np.arange(1, looks + 1) * m
    diff = np.cumsum(y_t.sum(2), 1) / n - np.cumsum(y_c.sum(2), 1) / n
    var = 2.0 / n
    fpr_seq = (msprt_pvalues(diff, var, 0.05) < 0.05).any(1).mean()
    fpr_naive = (naive_pvalues(diff, var) < 0.05).any(1).mean()
    assert fpr_seq <= 0.05
    assert fpr_naive > 0.15


def test_sequence_reports_first_rejection():
    n = np.arange(1, 21) * 1000.0
    diff = np.full(20, 0.2)
    out = msprt_sequence(diff, 1.0 / n, 0.04)
    assert out["first_reject_look"] is not None
    assert out["first_naive_reject_look"] <= out["first_reject_look"]
    null = msprt_sequence(np.zeros(20), 1.0 / n, 0.04)
    assert null["first_reject_look"] is None and null["first_naive_reject_look"] is None


def test_validation():
    with pytest.raises(ValueError):
        msprt_lambda(0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        tau2_from_effect(0.0)
    assert tau2_from_effect(0.1) == pytest.approx(0.01)
