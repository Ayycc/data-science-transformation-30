# Day 28: Legacy Reporting Modernisation: SAS/Excel to Iceberg + Ibis

**Tier IV - Distinguished** | **Domain:** Financial reporting & insurance | **Folder:** `day-28-legacy-reporting-modernisation-iceberg/`

**Core tools:** Apache Iceberg (pyiceberg), Ibis, DuckDB, Polars, datacompy

**Method:** Engine-portable transformations, open table formats, automated legacy-vs-modern reconciliation and performance benchmarking.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-28-legacy-reporting-modernisation-iceberg/`. Do not modify other days' folders.

## Business context

An insurer's quarterly regulatory report is produced by a chain of legacy SAS-style scripts and Excel macros that only two people understand. This is the transformation project leadership actually cares about.

## Data

Create a realistic 'legacy' implementation: procedural Python mimicking SAS DATA steps and Excel formulas (with known quirks: implicit rounding, merge-order dependence) over synthetic policy, premium and claims tables.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Inventory of the legacy logic as documented business rules with a dependency graph.
- Modern implementation with Ibis expressions (portable across DuckDB and Polars backends) writing Apache Iceberg tables via pyiceberg with a local catalog, including snapshots and schema evolution.
- Automated reconciliation suite (datacompy) comparing legacy and modern outputs row by row, with every intentional difference documented.
- Benchmark of runtime and cost proxies; a migration playbook covering parallel run, cut-over criteria and rollback.

## Acceptance criteria

- Reconciliation passes with zero unexplained differences
- Same Ibis code runs on two backends with identical results (test)
- Iceberg time-travel reproduces the previous quarter's report
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Column-level lineage from legacy rule to new model
- Cost model for cloud warehouse vs lakehouse

## Skills this project demonstrates

Data modernisation, Open table formats, Migration engineering

## Definition of done

1. Folder `day-28-legacy-reporting-modernisation-iceberg/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 28 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
