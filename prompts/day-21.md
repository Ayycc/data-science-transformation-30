# Day 21: Regulatory Q&A: Hybrid-Search RAG with Measured Faithfulness

**Tier III - Principal** | **Domain:** Government & regulation | **Folder:** `day-21-regulatory-rag-hybrid-search/`

**Core tools:** rank-bm25, sentence-transformers, FAISS or LanceDB, Anthropic SDK (optional), pandas

**Method:** Chunking strategies, hybrid lexical+dense retrieval, re-ranking, citation-grounded generation and retrieval/answer evaluation.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-21-regulatory-rag-hybrid-search/`. Do not modify other days' folders.

## Business context

Compliance analysts spend hours searching long regulations to answer questions. A chatbot is only acceptable if every answer cites the exact clause and hallucinations are measured.

## Data

Use public-domain regulatory text (for example sections of the US eCFR or EU GDPR text) if reachable; otherwise generate a synthetic regulation corpus. Build a gold Q&A set of at least 100 questions with supporting passages.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Ingestion with structure-aware chunking (sections/clauses) vs fixed-size chunking.
- BM25, dense, and hybrid (reciprocal rank fusion) retrieval with optional cross-encoder re-ranking.
- Answer generation with inline citations via Claude when a key is set; extractive fallback otherwise.
- Evaluation: recall@k, MRR, nDCG for retrieval; citation precision and faithfulness checks for answers.

## Acceptance criteria

- Hybrid retrieval beats both single methods on recall@5
- Every generated answer carries at least one valid citation
- Full eval runs offline with the fallback
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Query rewriting and HyDE
- Guardrail that refuses out-of-scope questions

## Skills this project demonstrates

Information retrieval, RAG, LLM evaluation

## Definition of done

1. Folder `day-21-regulatory-rag-hybrid-search/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 21 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
