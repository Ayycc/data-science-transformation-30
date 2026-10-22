import networkx as nx

from policy_causal.config import CONFOUNDERS
from policy_causal.dag import TREATMENT, adjustment_report, build_dag, dowhy_identification, is_valid_adjustment
from policy_causal.simulate import observed


def test_dag_is_acyclic_and_marks_unobserved():
    g = build_dag()
    assert nx.is_directed_acyclic_graph(g)
    assert g.nodes["ability"]["observed"] == "no"


def test_adjustment_sets():
    g = build_dag()
    assert is_valid_adjustment(g, TREATMENT, "earnings_post", set(CONFOUNDERS))
    assert not is_valid_adjustment(g, TREATMENT, "earnings_post", set(CONFOUNDERS) | {"certificate"})  # mediator / collider
    assert not is_valid_adjustment(g, TREATMENT, "earnings_post", set())
    assert not is_valid_adjustment(g, TREATMENT, "earnings_post", {"age", "educ"})
    rep = adjustment_report(CONFOUNDERS)
    assert "certificate" not in rep["earnings_post"]["minimal_set"]
    assert "distance_km" not in rep["earnings_post"]["minimal_set"]


def test_dowhy_agrees_with_networkx(people):
    r = dowhy_identification(observed(people).head(800))
    rep = adjustment_report(CONFOUNDERS)
    assert set(r["backdoor_variables"]) == set(rep["earnings_post"]["minimal_set"])
    assert r["instrumental_variables"] == ["distance_km"]
