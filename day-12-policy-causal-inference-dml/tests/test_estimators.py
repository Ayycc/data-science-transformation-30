import numpy as np

from policy_causal.config import CONFOUNDERS, MODIFIERS
from policy_causal.dml import dr_learner_ate, linear_dml_ate
from policy_causal.propensity import fit_propensity, ipw_estimate, match_pairs, matching_estimate, smd
from policy_causal.refute import evalue_continuous, evalue_from_risks, evalue_rr
from policy_causal.simulate import observed, true_effects


def test_ipw_balances_and_recovers(people):
    df = observed(people)
    x, t = df[CONFOUNDERS], df["treated"].to_numpy()
    e = fit_propensity(x, t)
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    assert smd(x, t).abs().max() > 0.3
    assert smd(x, t, w).abs().max() < 0.1
    truth = true_effects(people)["ate_earnings"]
    est = ipw_estimate(df["earnings_post"].to_numpy(float), t, e)
    assert abs(est - truth) < 700


def test_matching_closer_than_naive(people):
    df = observed(people)
    t = df["treated"].to_numpy()
    e = fit_propensity(df[CONFOUNDERS], t)
    pairs = match_pairs(t, e)
    y = df["earnings_post"].to_numpy(float)
    truth = true_effects(people)["ate_earnings"]
    naive = y[t == 1].mean() - y[t == 0].mean()
    assert abs(matching_estimate(y, t, pairs) - truth) < abs(naive - truth) / 3


def test_dml_ci_covers_truth(people):
    df = observed(people)
    ctrl = [c for c in CONFOUNDERS if c not in MODIFIERS]
    truth = true_effects(people)["ate_earnings"]
    r = linear_dml_ate(df, "earnings_post", MODIFIERS, ctrl, 3, 0, quick=True)
    assert r["ci_low"] <= truth <= r["ci_high"]
    d = dr_learner_ate(df, "earnings_post", CONFOUNDERS, 3, 0, quick=True)
    assert abs(d["estimate"] - truth) < 3 * d["se"]


def test_evalues():
    assert abs(evalue_rr(2.0) - (2 + np.sqrt(2))) < 1e-9
    assert evalue_rr(0.5) == evalue_rr(2.0)
    r = evalue_from_risks(0.6, 0.5, 0.05)
    assert r["evalue_point"] > r["evalue_ci"] > 1
    assert evalue_continuous(0.0, -1.0, 1.0)["evalue_ci"] == 1.0
