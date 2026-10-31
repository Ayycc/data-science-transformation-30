import numpy as np

from regrag.retrieval import BM25Retriever, DenseRetriever, Hit, HybridRetriever, rrf, tokenize


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The Complaints of customers, within 30 days") == ["complaint", "customer", "30", "day"]


def test_rrf_math_and_order():
    a = [Hit("x", 0, 0), Hit("y", 0, 1)]
    b = [Hit("y", 0, 0), Hit("z", 0, 1)]
    fused = rrf([a, b], k=60)
    assert [h.chunk_id for h in fused] == ["y", "x", "z"]
    assert np.isclose(fused[0].score, 1 / 61 + 1 / 62)


def test_lexical_question_hits_gold_clause(corpus, chunks):
    bm = BM25Retriever(chunks)
    lsa = DenseRetriever(chunks, backend="lsa")
    hy = HybridRetriever(bm, lsa)
    lex = [q for q in corpus["questions"] if q.style == "lexical"][:40]
    for r in (bm, lsa, hy):
        hits = [q.gold_clause_ids[0] in [h.chunk_id for h in r.search(q.question, 5)] for q in lex]
        assert np.mean(hits) >= 0.9, r.name


def test_dense_backends_and_index(chunks):
    d = DenseRetriever(chunks)  # static vectors if en_core_web_md is installed, else LSA
    assert d.backend in {"static", "lsa"}
    assert d.index.ntotal == len(chunks)
    e = d.embed(["customer complaints"])
    assert np.isclose(np.linalg.norm(e[0]), 1.0, atol=1e-4)
    s = d.scores("customer complaints")
    assert s.shape == (len(chunks),)


def test_prf_expands_query(chunks):
    bm = BM25Retriever(chunks)
    q = "civil penalty customer complaints"
    ex = bm.expand_query(q)
    assert ex[: len(tokenize(q))] == tokenize(q) and len(ex) > len(tokenize(q))
