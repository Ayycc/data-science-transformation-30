import numpy as np

from credit26 import fairness as F
from credit26 import mitigation as Mi


def _toy(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.choice(np.array(["A", "B", "C"]), n, p=[0.6, 0.25, 0.15])
    yg = (rng.random(n) < 0.85).astype(int)
    ap = (rng.random(n) < np.where(g == "A", 0.75, 0.55) + 0.1 * yg).astype(int)
    return ap, yg, g


def test_vectorised_metrics_match_fairlearn():
    ap, yg, g = _toy()
    pt = F.fairness_point(ap, yg, g, "A")
    levels = ["A", "B", "C"]
    m = F._metrics_from_counts(F.cell_counts(ap, yg, g, levels)[None], levels, 0)
    assert np.isclose(m["dp_difference"][0], pt["dp_difference"])
    assert np.isclose(m["eo_difference"][0], pt["eo_difference"])
    assert np.isclose(m["air_min"][0], pt["air_min"])
    sel = {k: ap[g == k].mean() for k in levels}
    assert np.isclose(pt["air"]["B"], sel["B"] / sel["A"])


def test_bootstrap_ci_covers_point():
    ap, yg, g = _toy(seed=3)
    a = F.audit(ap, yg, g, "A", n_boot=300, seed=1)
    lo, hi = a["air_min_ci"]
    assert lo <= a["air_min"] <= hi and hi - lo < 0.15
    assert a["passes_four_fifths"] == (a["air_min"] >= 0.8)


def test_proxy_strength_flags_geography(parts):
    tr = parts.train
    s = F.proxy_strength(tr, ["zip_cluster", "employment_years"], ["zip_cluster"], "group", "A")
    assert s["zip_cluster"] > 0.75 and s["employment_years"] < 0.6


def test_reweighing_balances_label_rates():
    rng = np.random.default_rng(0)
    g = rng.choice(np.array(["A", "B"]), 5000, p=[0.7, 0.3])
    y = (rng.random(5000) < np.where(g == "A", 0.1, 0.3)).astype(int)
    w = Mi.reweighing_weights(g, y, 1.0)
    rates = [np.average(y[g == k], weights=w[g == k]) for k in ["A", "B"]]
    assert abs(rates[0] - rates[1]) < 1e-9
    assert np.allclose(Mi.reweighing_weights(g, y, 0.0), 1.0)


def test_threshold_optimizer_equalises_selection(parts, mono_model):
    to = Mi.threshold_optimizer(mono_model.p_bad, parts.valid, "group")
    ap = Mi.to_approve(to, parts.test, "group")
    pt = F.fairness_point(ap, 1 - parts.test["bad"].to_numpy(), parts.test["group"].to_numpy(), "A")
    assert pt["air_min"] > 0.85
