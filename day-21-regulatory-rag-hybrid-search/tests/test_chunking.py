import pytest

from regrag.chunking import fixed_chunks, relevant_chunk_ids


def test_structure_one_chunk_per_clause(corpus, chunks):
    assert len(chunks) == len(corpus["clauses"])
    for ch in chunks:
        assert ch.clause_ids == [ch.chunk_id]
        assert ch.text.startswith("§ ")


def test_fixed_windows_overlap_and_cover(corpus):
    fx = fixed_chunks(corpus["clauses"], size=40, overlap=10)
    assert all(len(c.text.split()) <= 40 for c in fx)
    assert all(fx[i].end > fx[i + 1].start for i in range(len(fx) - 1))  # overlapping
    assert fx[-1].end == len(corpus["doc"].rstrip("\n"))
    with pytest.raises(ValueError):
        fixed_chunks(corpus["clauses"], size=10, overlap=10)


def test_relevance_contains_answer_span(corpus, chunks):
    q = next(q for q in corpus["questions"] if q.in_scope)
    rel = relevant_chunk_ids(chunks, q.gold_clause_ids, corpus["spans"])
    assert rel == set(q.gold_clause_ids)
    fx = fixed_chunks(corpus["clauses"], size=40, overlap=10)
    rel_fx = relevant_chunk_ids(fx, q.gold_clause_ids, corpus["spans"])
    a, b = corpus["spans"][q.gold_clause_ids[0]]
    assert rel_fx and all(c.start <= a and b <= c.end for c in fx if c.chunk_id in rel_fx)
