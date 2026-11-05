import numpy as np
import pandas as pd

from credit26.data import SYNTH_SPEC
from credit26.models import GBMConfig, cutoff_for_approval, psi, psi_categorical, train_gbm
from credit26.pipeline import monotonicity_check
from credit26.scorecard import Scorecard


def test_psi_properties():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    assert psi(x, x) < 1e-9
    assert psi(x, rng.normal(size=20000)) < 0.01
    assert psi(x, rng.normal(0.5, 1, size=20000)) > 0.2
    s = pd.Series(list("aabbc") * 100)
    assert psi_categorical(s, s) < 1e-9


def test_cutoff_hits_approval_rate():
    p = np.random.default_rng(1).random(10000)
    c = cutoff_for_approval(p, 0.7)
    assert abs((p <= c).mean() - 0.7) < 0.001


def test_scorecard_woe_and_points(parts):
    tr = parts.train
    sc = Scorecard(SYNTH_SPEC.numeric, SYNTH_SPEC.categorical).fit(tr, tr["bad"].to_numpy())
    w = sc.binner.woe["credit_utilization"]
    # WOE = ln(%good/%bad): the lowest-utilisation bin is the safest
    assert w[min(w)] > w[max(w)]
    assert all(v >= 0 for v in sc.binner.iv.values()) and sc.binner.iv["credit_utilization"] > 0.1
    te = parts.test
    score, p = sc.score(te), sc.predict_proba_bad(te)
    # points are a monotone decreasing transform of p_bad: score = offset + factor * log-odds(good)
    expected = sc.offset + sc.factor * np.log((1 - p) / p)
    assert np.allclose(score, expected, atol=1e-6)
    rc = sc.reason_codes(te.head(50))
    assert rc.shape == (50, 4) and all(len(set(r)) == 4 for r in rc)


def test_monotone_constraints_hold(parts, mono_model):
    chk = monotonicity_check(mono_model, parts.test, SYNTH_SPEC, n=40, seed=0)
    assert min(chk.values()) == 1.0


def test_gbm_learns(parts):
    from sklearn.metrics import roc_auc_score

    m = train_gbm("g", parts.train, SYNTH_SPEC.features, SYNTH_SPEC, GBMConfig(n_estimators=80))
    assert roc_auc_score(parts.test["bad"], m.p_bad(parts.test)) > 0.68
