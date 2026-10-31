import random

from regrag.evaluate import (
    answer_correct,
    check_faithfulness,
    corrupt,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
    split_claims,
    stress_test,
)
from regrag.generate import Answer

CTX = {
    "§2.1(c)": "A regulated entity shall notify the regulator of any data breach affecting customer records within 30 business days of discovery of the incident."
}
GOOD = 'Answer: 30 business days [§2.1(c)]. Source: "A regulated entity shall notify the regulator of any data breach affecting customer records within 30 business days of discovery of the incident." [§2.1(c)].'


def test_retrieval_metrics():
    assert recall_at_k(["a", "b", "c"], {"c"}, 2) == 0.0 and recall_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert mrr_at_k(["a", "b"], {"b"}) == 0.5
    assert abs(ndcg_at_k(["a", "b"], {"b"}) - 1 / 1.5849625) < 1e-6
    bs = paired_bootstrap([1, 1, 0, 1], [0, 1, 0, 1])
    assert bs["diff"] == 0.25 and bs["ci_low"] <= 0.25 <= bs["ci_high"]


def test_split_claims_and_uncited_tail():
    segs = split_claims(GOOD + " Also firms get an extension.")
    assert [c for _, c in segs] == [["§2.1(c)"], ["§2.1(c)"], []]


def test_faithful_answer_passes():
    assert all(c.supported for c in check_faithfulness(GOOD, CTX))


def test_checker_catches_hallucinations():
    bad_num = GOOD.replace("Answer: 30", "Answer: 45")
    assert check_faithfulness(bad_num, CTX)[0].reason == "number_not_in_source"
    bad_cite = GOOD.replace("[§2.1(c)]", "[§9.9(z)]", 1)
    assert check_faithfulness(bad_cite, CTX)[0].reason == "citation_not_in_context"
    added = GOOD + " Firms may apply for an extension in exceptional circumstances [§2.1(c)]."
    assert not check_faithfulness(added, CTX)[-1].supported
    assert not check_faithfulness("Answer: 30 business days.", CTX)[0].supported  # uncited


def test_answer_correct():
    assert answer_correct("the board audit committee, at least quarterly", "the board audit committee, quarterly")
    assert not answer_correct("the Chief Risk Officer, at least quarterly", "the board audit committee, quarterly")
    assert answer_correct("up to $50,000 per violation", "$50,000")


def test_stress_test_detects_corruptions():
    a = Answer(
        "Q1",
        GOOD,
        ["§2.1(c)"],
        False,
        {**CTX, "§2.1(d)": "All customer records shall be reviewed by the Chief Risk Officer at least annually."},
        "extractive",
        0.9,
    )
    assert corrupt(a, "fact_swap", random.Random(0), None) != GOOD
    res = stress_test([a])
    assert res["clean"]["false_alarm_rate"] == 0.0
    assert res["fact_swap"]["detection_rate"] == 1.0 and res["added_claim"]["detection_rate"] == 1.0
    assert res["citation_swap"]["detection_rate"] == 1.0
