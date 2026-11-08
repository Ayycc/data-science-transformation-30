from semsql.agent import RawRulesAgent, RulesAgent
from semsql.benchmark import (
    evaluate,
    gold_answers,
    load_questions,
    results_match,
    summarise,
)


def test_results_match():
    assert results_match([[1, "a"]], [["a", 1.004]])[0]  # column permutation + tolerance
    assert not results_match([[1, "a"]], [["a", 1.5]])[0]
    assert not results_match([[1], [2]], [[1]])[0]
    assert results_match([["x", 1], ["y", 2]], [["y", 2], ["x", 1]])[0]
    assert not results_match([["x", 1], ["y", 2]], [["y", 2], ["x", 1]], ordered=True)[0]
    assert results_match([[None]], [[None]])[0]


def test_benchmark_shape():
    qs = load_questions()
    assert len(qs) == 60 and len({q["id"] for q in qs}) == 60
    assert sum(q["expected"] == "answer" for q in qs) == 47
    assert all(q.get("gold_sql") for q in qs if q["expected"] == "answer")


def test_benchmark_offline_semantic_beats_raw(toolbox, con):
    qs = load_questions()
    gold = gold_answers(con, qs)
    sem = summarise(evaluate(RulesAgent(toolbox), qs, gold))
    raw = summarise(evaluate(RawRulesAgent(toolbox), qs, gold))
    assert sem["execution_accuracy"] >= 0.9
    assert sem["refusal_correctness"] == 1.0 and sem["unsafe_answers"] == 0
    assert raw["execution_accuracy"] < sem["execution_accuracy"]


def test_answer_cache(toolbox):
    agent = RulesAgent(toolbox)
    a1 = agent.ask("Revenue by region in FY2025")
    a2 = agent.ask("revenue by region in fy2025")
    assert not a1.cached and a2.cached and a1.rows == a2.rows and toolbox.cache_hits == 1
