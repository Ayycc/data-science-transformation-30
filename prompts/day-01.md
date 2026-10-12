# Day 01: Retail Medallion Lakehouse with dbt + DuckDB

**Tier I - Advanced** | **Domain:** Retail | **Folder:** `day-01-retail-medallion-lakehouse/`

**Core tools:** DuckDB, dbt-duckdb, SQL, Parquet, Python

**Method:** Medallion (bronze/silver/gold) ELT, dimensional modelling, SCD Type 2 snapshots, dbt tests and docs.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-01-retail-medallion-lakehouse/`. Do not modify other days' folders.

## Business context

A mid-size omnichannel retailer runs reporting off spreadsheets exported from its POS and e-commerce systems. Numbers disagree between teams. Build the analytical backbone that becomes the single source of truth.

## Data

Generate a realistic synthetic dataset (2 years, ~2M order lines, ~50k customers, ~5k SKUs, 120 stores + web) with a seeded Python generator using Faker/NumPy. Inject realistic mess: duplicate events, late-arriving records, timezone drift, price changes, returns, renamed SKUs. Optionally enrich with the public UCI Online Retail II dataset if reachable.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Bronze layer: raw Parquet landing written by the generator, loaded into DuckDB as sources.
- Silver layer: dbt staging + intermediate models with dedup, typing, timezone normalisation, and late-arrival handling.
- Gold layer: star schema (fct_order_lines, fct_returns, dim_customer, dim_product as SCD2 via dbt snapshots, dim_store, dim_date) plus marts for daily sales, cohort retention and basket metrics.
- dbt tests (unique, not_null, relationships, accepted_values, custom singular tests for revenue reconciliation) and dbt docs with exposures.
- An analysis notebook or script that answers 5 business questions from the gold layer with charts saved to results/.

## Acceptance criteria

- `dbt build` passes end-to-end against a fresh DuckDB file
- Revenue in gold reconciles to bronze within 0.01% after documented exclusions
- SCD2 history is demonstrably correct for at least one renamed SKU
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Incremental models with merge strategy
- A data-quality scorecard model

## Skills this project demonstrates

Dimensional modelling, dbt, Data quality testing, Analytical SQL

## Definition of done

1. Folder `day-01-retail-medallion-lakehouse/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 01 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
