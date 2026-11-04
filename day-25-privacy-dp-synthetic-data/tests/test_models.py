import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from privacy25.data import CATEGORICAL, COLUMNS, NUMERIC_BOUNDS, PopulationConfig, generate_population, parse_adult
from privacy25.dp_model import ObjectivePerturbationLR
from privacy25.encoding import discretize, linear_features, undiscretize
from privacy25.evaluate import dcr_membership_inference, marginal_fidelity, qi_uniqueness
from privacy25.sdc import k_anonymise
from privacy25.synth.copula import GaussianCopula
from privacy25.synth.ctgan_np import CTGANLite
from privacy25.synth.mst import DPTreeSynthesizer

ADULT = """39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, Mexico, >50K.
38, ?, 215646, HS-grad, 9, Divorced, ?, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K
"""


def _valid(df: pd.DataFrame) -> None:
    assert list(df.columns) == COLUMNS
    for c, lv in CATEGORICAL.items():
        assert df[c].isin(lv).all(), c
    for c, (lo, hi) in NUMERIC_BOUNDS.items():
        assert df[c].between(lo, hi).all(), c


def test_generator_seeded_schema_and_rare_group(pop):
    _valid(pop)
    assert pop.equals(generate_population(PopulationConfig(n=4000, seed=7)))
    share = (pop["race"] == "Amer-Indian-Eskimo").mean()
    assert 0.004 < share < 0.02
    assert 0.18 < (pop["income"] == ">50K").mean() < 0.30


def test_parse_adult_fixture():
    df = parse_adult(ADULT)
    assert len(df) == 2  # the row with '?' is dropped
    assert df.loc[1, "native_region"] == "Latin-America" and df.loc[1, "income"] == ">50K"
    _valid(df)


def test_discretize_roundtrip(pop):
    codes = discretize(pop)
    back = undiscretize(codes, np.random.default_rng(0))
    _valid(back)
    assert (discretize(back) == codes).all()


def test_objective_perturbation_privacy_terms_and_utility(parts):
    tr, te = parts
    X, y, dn = linear_features(tr)
    Xt, yt, _ = linear_features(te)
    assert np.linalg.norm(X, axis=1).max() <= dn + 1e-9
    hi = ObjectivePerturbationLR(1e6, dn, 1e-4, 0).fit(X, y)
    lo = ObjectivePerturbationLR(0.01, dn, 1e-4, 0).fit(X, y)
    auc_hi = roc_auc_score(yt, hi.predict_proba(Xt)[:, 1])
    assert auc_hi > 0.8
    assert hi.noise_norm_ < lo.noise_norm_
    eps_p, delta = ObjectivePerturbationLR(1.0, dn, 1e-4).privacy_terms(len(y))
    assert 0 < eps_p <= 1.0 and delta >= 0


def test_copula_and_ctgan_produce_valid_microdata(parts):
    tr, _ = parts
    _valid(GaussianCopula(seed=0).fit(tr).sample(500))
    _valid(CTGANLite(epochs=1, seed=0).fit(tr).sample(500))


def test_dp_tree_stays_in_budget_and_is_a_tree(parts):
    tr, _ = parts
    m = DPTreeSynthesizer(1.0, 1e-6, seed=0).fit(tr)
    assert m.epsilon_spent_ <= 1.0 + 1e-9
    assert len(m.edges_) == len(COLUMNS) - 1
    nodes = {a for e in m.edges_ for a in e}
    assert nodes == set(range(len(COLUMNS)))
    s = m.sample()
    _valid(s)
    assert abs(len(s) - len(tr)) < 0.05 * len(tr)


def test_k_anonymise_reaches_k(parts):
    tr, _ = parts
    rel, info = k_anonymise(tr, 5)
    assert info["after"]["min_k"] >= 5 and info["suppressed_records"] + len(rel) == len(tr)


def test_risk_metrics_detect_a_copy(parts):
    tr, te = parts
    copy = dcr_membership_inference(tr, te, tr, 300, 0)
    fresh = dcr_membership_inference(tr, te, generate_population(PopulationConfig(n=3000, seed=99)), 300, 0)
    assert copy["auc"] > 0.65 and copy["exact_match_rate_members"] == 1.0
    assert abs(fresh["auc"] - 0.5) < 0.1
    assert marginal_fidelity(tr, tr)["tvd_2way_mean"] == 0
    q = qi_uniqueness(tr, te, tr)
    assert q["release_rows_matching_train_unique"] > q["holdout_rows_matching_train_unique"]
