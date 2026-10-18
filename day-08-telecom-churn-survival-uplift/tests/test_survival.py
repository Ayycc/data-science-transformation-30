from __future__ import annotations

import numpy as np

from retention import survival as sv


def test_treated_customers_censored_at_campaign(small):
    panel, _ = small
    sf = sv.survival_frame(panel)
    tr = (panel["treatment"] == 1).to_numpy()
    assert (sf["duration"].to_numpy()[tr] <= panel["tenure_at_campaign"].to_numpy()[tr]).all()
    pre = sv.survival_frame(panel, calendar_cutoff=30)
    assert (pre["duration"] <= 30 - panel.loc[pre.index, "start_month"]).all()


def test_km_and_logrank(small):
    sf = sv.survival_frame(small[0])
    curves, p = sv.kaplan_meier_by(sf, "contract", sv.CONTRACT_LABELS)
    assert p < 1e-10
    tab = sv.km_table(curves, [12])
    s = dict(zip(tab["segment"], tab["S(12)"]))
    assert s["month_to_month"] < s["one_year"] < s["two_year"]


def test_ph_violation_detected_and_fixed_by_stratification(small):
    sf = sv.survival_frame(small[0])
    cox = sv.fit_cox(sf, stratified=False)
    ph = sv.ph_test(cox, sf, stratified=False).set_index("covariate")["p_value"]
    assert ph["contract_one_year"] < 1e-4 and ph["contract_two_year"] < 1e-4
    cox_s = sv.fit_cox(sf, stratified=True)
    ph_s = sv.ph_test(cox_s, sf, stratified=True)
    assert not any(c.startswith("contract") for c in ph_s["covariate"])
    assert ph_s["p_value"].min() > 0.001


def test_models_discriminate(small):
    sf = sv.survival_frame(small[0])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(sf))
    te, tr = sf.iloc[idx[:4000]], sf.iloc[idx[4000:]]
    times = np.array([6, 12, 24])
    cox_s = sv.fit_cox(tr, stratified=True)
    S = sv.cox_survival_matrix(cox_s, te, times)
    assert S.shape == (len(te), 3) and (np.diff(S, axis=1) <= 1e-9).all()
    m = sv.evaluate(tr, te, S, times, 12)
    assert 0.7 < m["harrell_c"] < 0.9 and 0.7 < m["td_auc_mean"] < 0.95
    rsf = sv.fit_rsf(tr.iloc[:3000], 20, 40, 0)
    m2 = sv.evaluate(tr, te, sv.rsf_survival_matrix(rsf, te, times), times, 12)
    assert m2["harrell_c"] > 0.7


def test_aalen_johansen_partitions_churn(small):
    sf = sv.survival_frame(small[0])
    aj = sv.aalen_johansen(sf, 36)
    total = aj[[c for c in aj.columns if c.startswith("cif_")]].sum(1)
    assert np.allclose(total, 1 - aj["overall_survival"])
    assert (aj["cif_port_out"].diff().dropna() >= 0).all()
