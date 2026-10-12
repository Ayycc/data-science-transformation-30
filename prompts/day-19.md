# Day 19: Data Mesh in Practice: Contracts, Versioning and Lineage

**Tier III - Principal** | **Domain:** Enterprise data platform | **Folder:** `day-19-data-mesh-contracts-lineage/`

**Core tools:** dbt (model contracts), OpenLineage, JSON Schema, DuckDB, GitHub Actions

**Method:** Domain-owned data products, enforceable contracts, semantic versioning, breaking-change detection and column-level lineage.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-19-data-mesh-contracts-lineage/`. Do not modify other days' folders.

## Business context

A company decentralised data ownership to domains (orders, payments, logistics). Downstream dashboards break weekly when a domain renames a column.

## Data

Synthetic multi-domain e-commerce data produced by three independent 'domain' generators with their own release cadence and schema evolution.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Three data products, each with a contract file (schema, SLOs, owner, semantic version) in the open data contract standard style.
- dbt projects using model contracts and versions; a consumer project composing them into a cross-domain mart.
- A CI check script that diffs contracts and fails on breaking changes without a major version bump.
- Lineage emitted as OpenLineage events and rendered as a column-level lineage graph.
- A short architecture decision record (ADR) on governance trade-offs.

## Acceptance criteria

- A deliberate breaking change is caught by CI (test)
- Lineage graph traces a gold metric back to source columns
- Each data product reports its SLO status
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Data product catalogue page generated from contracts
- Deprecation workflow with dual-publishing

## Skills this project demonstrates

Data architecture, Governance, Platform engineering

## Definition of done

1. Folder `day-19-data-mesh-contracts-lineage/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 19 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
