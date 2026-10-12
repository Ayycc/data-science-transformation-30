# Day 05: City 311 Data Quality and Service Hotspots

**Tier I - Advanced** | **Domain:** Public sector | **Folder:** `day-05-civic-311-data-quality-hotspots/`

**Core tools:** pandera, Great Expectations, H3, GeoPandas, DuckDB

**Method:** Data contracts and validation suites, anomaly detection on operational KPIs, hexagonal spatial aggregation.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-05-civic-311-data-quality-hotspots/`. Do not modify other days' folders.

## Business context

A city wants to publish service-level dashboards for 311 requests, but the data is riddled with missing coordinates, impossible timestamps and inconsistent categories. Trust has to come before dashboards.

## Data

Use a public 311 open-data extract (NYC, Chicago or SF Socrata APIs) if reachable; otherwise generate ~1M synthetic requests across a real city's bounding box with realistic defects.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- A pandera schema and a Great Expectations suite describing the data contract, with a validation report artifact.
- A quarantine pattern: failing rows routed to a reject table with reason codes; quality KPIs over time.
- Category harmonisation with fuzzy matching and a reviewed mapping table.
- H3-based hotspot analysis of resolution times with Getis-Ord style or z-score hotspots, mapped to an HTML file.

## Acceptance criteria

- Every defect class injected or found is caught by at least one check
- Quarantined rows are fully traceable to a rule
- Hotspot map renders offline from results/
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Equity analysis of resolution time by neighbourhood demographics
- Data-quality SLO burn-down chart

## Skills this project demonstrates

Data contracts, Data validation, Geospatial analytics

## Definition of done

1. Folder `day-05-civic-311-data-quality-hotspots/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 05 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
