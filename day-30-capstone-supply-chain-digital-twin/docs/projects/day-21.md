# Day 21: Regulatory Q&A: Hybrid-Search RAG with Measured Faithfulness

**Tier III - Principal** | **Domain:** Government & regulation | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-21-regulatory-rag-hybrid-search/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-21-regulatory-rag-hybrid-search/README.md)

- **Key metric:** recall@5 hybrid + learned re-ranker: **81.2%** (BM25 recall@5 68.0%)
- **Data:** synthetic fallback (public source blocked); public sources attempted: eCFR API; reachable: none
- **Tests (static):** 27 functions, 27 cases (27 stated in README)
- **Lines of code:** 1,761 Python source, 257 tests, 0 SQL
- **CI:** success
- **Tools:** anthropic, en-core-web-md, faiss-cpu, matplotlib, numpy, rank-bm25, scikit-learn, scipy, spacy

## Acceptance criteria not met

- hybrid search beats both BM25 and dense alone on recall@5: met for structure-aware chunks (not significant vs dense); fixed-size chunks met only narrowly with weighted RRF (0.656 vs BM25 0.652, not significant), plain RRF 0.648

## Caveats

- No live LLM run; extractive offline answers (faithfulness 100% by construction)
