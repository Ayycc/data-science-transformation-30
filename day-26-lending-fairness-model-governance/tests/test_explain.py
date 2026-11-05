import numpy as np

from credit26 import explain as X
from credit26.data import SYNTH_SPEC
from credit26.models import cutoff_for_approval


def test_shap_additivity_and_four_reasons(parts, mono_model):
    te = parts.test
    sv, base = X.shap_values(mono_model, te)
    raw = mono_model.model.predict(mono_model.X(te), raw_score=True)
    assert np.max(np.abs(sv.sum(1) + base - raw)) < 1e-6
    names, vals = X.reason_codes(sv, mono_model.features, k=4)
    assert names.shape == (len(te), 4)
    assert np.all(np.diff(vals, axis=1) <= 1e-12)  # ranked
    rf = X.reason_frame(te, names, vals, SYNTH_SPEC, mono_model.features)
    assert rf[[f"reason_{j}_code" for j in range(1, 5)]].notna().all().all()


def test_counterfactuals_are_valid(parts, mono_model):
    cut = cutoff_for_approval(mono_model.p_bad(parts.valid), 0.7)
    p = mono_model.p_bad(parts.test)
    dec = parts.test[p > cut].head(25).reset_index(drop=True)
    q = parts.train[list(SYNTH_SPEC.actionable)].quantile([0.25, 0.75])
    scale = {f: float(q.loc[0.75, f] - q.loc[0.25, f]) for f in SYNTH_SPEC.actionable}
    cf = X.counterfactuals(mono_model, dec, cut, SYNTH_SPEC, scale, K=6)
    assert len(cf) == 25 and cf["found"].mean() > 0.8
    for _, r in cf[cf["found"]].iterrows():
        assert r["p_bad_after"] <= cut < r["p_bad_before"]
        row = dec[dec["application_id"] == r["application_id"]].iloc[0].copy()
        for ch in r["changes"].split("; "):
            f, vals = ch.split(": ")
            assert f in SYNTH_SPEC.actionable
            a, b = (float(v.replace(",", "")) for v in vals.split(" -> "))
            d = SYNTH_SPEC.actionable[f][0]
            assert (b - a) * d > 0  # moved only in the allowed direction
            assert abs(a - row[f]) <= 0.01 * max(abs(row[f]), 1)  # "from" value is the applicant's own
