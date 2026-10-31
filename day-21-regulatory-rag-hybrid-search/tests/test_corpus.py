from regrag.corpus import CLAUSE_TYPES, TITLE, generate_corpus, generate_questions


def test_deterministic(corpus):
    again = generate_corpus()
    assert [c.text for c in again] == [c.text for c in corpus["clauses"]]
    assert [q.question for q in generate_questions(again)] == [q.question for q in corpus["questions"]]


def test_gold_set_size_and_integrity(corpus):
    qs, by_id = corpus["questions"], corpus["by_id"]
    in_scope = [q for q in qs if q.in_scope]
    assert len(in_scope) >= 100
    assert len({c.clause_id for c in corpus["clauses"]}) == len(corpus["clauses"])
    for q in in_scope:
        (cid,) = q.gold_clause_ids
        clause = by_id[cid]
        assert clause.clause_type in CLAUSE_TYPES
        assert q.answer.split(", ")[0].replace("the ", "", 1) in clause.text
    assert any(not q.in_scope for q in qs)


def test_split_by_section_has_no_leakage(corpus):
    sec = {q.qid: q.gold_clause_ids[0].split("(")[0] for q in corpus["questions"] if q.in_scope}
    dev = {sec[q.qid] for q in corpus["questions"] if q.in_scope and q.split == "dev"}
    test = {sec[q.qid] for q in corpus["questions"] if q.in_scope and q.split == "test"}
    assert dev and test and not dev & test


def test_paraphrase_differs_from_lexical(corpus):
    qs = [q for q in corpus["questions"] if q.in_scope]
    pairs = list(zip(qs[0::2], qs[1::2], strict=True))
    assert all(a.style == "lexical" and b.style == "paraphrase" for a, b in pairs)
    assert sum(a.question != b.question for a, b in pairs) == len(pairs)


def test_labelled_synthetic():
    assert "SYNTHETIC" in TITLE
