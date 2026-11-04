import pytest
from diffprivlib.mechanisms import Laplace

from privacy25.accounting import BudgetExceededError, PrivacyAccountant
from privacy25.data import CATEGORICAL
from privacy25.query_server import QueryServer
from privacy25.release import ReleasePlan, accuracy_curves, composition_comparison, contingency, laplace_scale, official_release


def test_laplace_scale_matches_diffprivlib():
    for eps in (0.1, 1.0, 3.0):
        assert Laplace(epsilon=eps, sensitivity=1.0).variance(0) == pytest.approx(2 * laplace_scale(1.0, eps) ** 2)


def test_contingency_covers_full_public_domain(pop):
    t = contingency(pop, ["race", "sex", "income"])
    assert len(t) == len(CATEGORICAL["race"]) * 2 * 2
    assert t.sum() == len(pop)


def test_official_release_spends_exactly_its_plan(pop):
    acc = PrivacyAccountant(2.0)
    out = official_release(pop, acc, ReleasePlan(epsilon=1.0), seed=3)
    assert acc.spent_epsilon == pytest.approx(1.0)
    assert out["budget"]["epsilon_spent"] == pytest.approx(1.0)
    assert out["tables"]["T2_age_histogram"]["tvd"] < 0.05
    with pytest.raises(BudgetExceededError):
        official_release(pop, acc, ReleasePlan(epsilon=1.5), seed=3)


def test_errors_shrink_with_epsilon_and_rare_cells_suffer_more(pop):
    c = accuracy_curves(pop, [0.1, 1.0, 10.0], trials=200, seed=0)
    for k in ("count_rel_err_white", "count_rel_err_rare", "age_hist_tvd"):
        assert c[k][0] > c[k][1] > c[k][2]
    assert all(r > w for r, w in zip(c["count_rel_err_rare"], c["count_rel_err_white"]))


def test_zcdp_beats_basic_composition_for_many_queries():
    comp = composition_comparison(1.0, 1e-6, [1, 10, 100])
    assert comp["laplace_std"][0] < comp["gaussian_zcdp_std"][0]  # one query: Laplace wins
    assert comp["laplace_std"][2] > comp["gaussian_zcdp_std"][2]  # many queries: zCDP wins


def test_query_server_enforces_budget_and_caches(pop):
    server = QueryServer(pop, PrivacyAccountant(0.3), seed=0)
    a1 = server.count({"race": "White"}, 0.1)
    a2 = server.count({"race": "White"}, 0.1)
    assert a2.cached and a2.value == a1.value and a2.epsilon_charged == 0
    server.mean("age", {"sex": "Male"}, 0.1)
    server.count({"race": "Black"}, 0.1)
    refused = server.count({"race": "Other"}, 0.1)
    assert refused.refused and refused.value is None
    assert server.accountant.spent_epsilon <= 0.3 + 1e-12
    assert 17 <= server.log[2].value <= 90
    with pytest.raises(ValueError):
        server.count({"race": "Martian"}, 0.1)
