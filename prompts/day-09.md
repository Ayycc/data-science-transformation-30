# Day 09: Clickstream Sessionisation at Scale with PySpark + Delta Lake

**Tier II - Expert** | **Domain:** Media & e-commerce | **Folder:** `day-09-clickstream-spark-delta/`

**Core tools:** PySpark, Delta Lake, Spark SQL, pytest, chispa

**Method:** Distributed ETL with window functions, skew mitigation, MERGE upserts, time travel and performance tuning.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-09-clickstream-spark-delta/`. Do not modify other days' folders.

## Business context

An online publisher's event volume outgrew single-node tooling. They need sessionised, deduplicated engagement tables refreshed incrementally.

## Data

Generate ~50M synthetic events (page views, clicks, scrolls, ad impressions) with bots, clock skew, duplicate deliveries and a heavy-hitter skew on a few viral articles. Run Spark in local mode.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Bronze to silver to gold Delta tables with schema enforcement and evolution.
- Sessionisation with 30-minute inactivity windows using window functions; bot filtering heuristics.
- Incremental MERGE for late-arriving events, idempotent reruns, and a time-travel audit query.
- Skew handling (salting or AQE skew join) with before/after Spark UI metrics or explain plans captured in docs.
- chispa/pytest unit tests for transformations.

## Acceptance criteria

- Rerunning a batch produces identical gold tables (idempotency test)
- Skew mitigation shows measurable runtime improvement
- OPTIMIZE/Z-ORDER or partitioning strategy is justified
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Structured Streaming version of the silver layer
- Change data feed consumers

## Skills this project demonstrates

Apache Spark, Lakehouse table formats, Performance tuning

## Definition of done

1. Folder `day-09-clickstream-spark-delta/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 09 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
