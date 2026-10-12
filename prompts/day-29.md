# Day 29: Agentic Analytics: Text-to-SQL over a Semantic Layer with Guardrails

**Tier IV - Distinguished** | **Domain:** Enterprise business intelligence | **Folder:** `day-29-agentic-text-to-sql-semantic-layer/`

**Core tools:** Anthropic SDK (optional), DuckDB, sqlglot, YAML semantic layer, pytest

**Method:** Semantic layer modelling, tool-using LLM agent, SQL validation and guardrails, execution-accuracy benchmarking.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-29-agentic-text-to-sql-semantic-layer/`. Do not modify other days' folders.

## Business context

Executives want to ask questions of company data in plain English. Previous chatbots produced confident wrong numbers because they guessed at raw tables.

## Data

Reuse a synthetic B2B SaaS warehouse (accounts, subscriptions, invoices, usage, support) generated in DuckDB with tricky semantics: fiscal calendars, churn definitions, currency conversion.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- A YAML semantic layer (metrics, dimensions, joins, definitions) compiled to SQL by a small compiler.
- An agent that plans, calls tools (list metrics, describe dimension, compile query, run query, check result) and answers with the SQL and caveats; uses Claude when ANTHROPIC_API_KEY is set, with a deterministic rules-based planner fallback.
- Guardrails: sqlglot parsing, read-only enforcement, row limits, PII column blocking, and refusal of undefined metrics.
- Benchmark of 60 questions with gold SQL/answers: execution accuracy, refusal correctness, and failure analysis, comparing semantic-layer vs raw-schema prompting.

## Acceptance criteria

- No write or DDL statement can execute (tests)
- Benchmark runs offline with the fallback and reports accuracy
- Semantic-layer approach measurably beats raw-schema approach when a key is present (or the comparison is documented as pending)
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Multi-turn clarifying questions
- Answer caching keyed on compiled SQL

## Skills this project demonstrates

LLM agents, Semantic modelling, AI safety guardrails

## Definition of done

1. Folder `day-29-agentic-text-to-sql-semantic-layer/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 29 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
