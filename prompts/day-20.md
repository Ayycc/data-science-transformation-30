# Day 20: LLM Structured Extraction with a Rigorous Evaluation Harness

**Tier III - Principal** | **Domain:** Legal & procurement | **Folder:** `day-20-llm-document-extraction-eval/`

**Core tools:** Anthropic SDK (optional), Pydantic, pdfplumber, rapidfuzz, pandas

**Method:** Schema-constrained LLM extraction, deterministic baselines, gold-set evaluation, cost/latency tracking.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-20-llm-document-extraction-eval/`. Do not modify other days' folders.

## Business context

A procurement team manually keys data from thousands of supplier contracts and invoices. They want automation, but only with measured accuracy per field.

## Data

Generate ~500 synthetic contracts and invoices as PDFs and text (varied templates, clauses, tables, scanned-style noise) with gold JSON labels.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Pydantic target schemas (parties, dates, amounts, payment terms, renewal clauses, line items).
- Extractor interface with: rule/regex baseline, and an LLM extractor using Claude via the Anthropic SDK when ANTHROPIC_API_KEY is set (tool-use / structured output), with retries and validation-repair loop.
- Evaluation harness: per-field exact/fuzzy match, line-item F1, calibration of confidence, and cost/latency per document.
- Error analysis report with failure taxonomy; human-in-the-loop routing rule for low-confidence fields.

## Acceptance criteria

- Harness runs fully offline with the baseline when no API key is present
- Per-field metrics table committed in results/
- Invalid outputs are never written downstream (validation tests)
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Prompt/version registry with regression tests
- Active sampling of documents for human review

## Skills this project demonstrates

LLM engineering, Evaluation design, Document AI

## Definition of done

1. Folder `day-20-llm-document-extraction-eval/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 20 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
