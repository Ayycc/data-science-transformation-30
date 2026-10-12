# Day 10: Air Quality Intelligence with Dagster Software-Defined Assets

**Tier II - Expert** | **Domain:** Environment & climate | **Folder:** `day-10-air-quality-dagster-assets/`

**Core tools:** Dagster, DuckDB, pandas, Pandera, Plotly

**Method:** Asset-oriented orchestration with partitions, sensors, asset checks, retries and observability.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-10-air-quality-dagster-assets/`. Do not modify other days' folders.

## Business context

An environmental agency ingests sensor readings from hundreds of stations and must publish daily pollution alerts. Today it is a fragile chain of cron scripts.

## Data

Use the OpenAQ API if reachable; otherwise simulate 300 stations reporting PM2.5, NO2 and O3 hourly with outages, drift and calibration offsets.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Dagster project with daily-partitioned assets: raw readings, cleaned readings, station health, daily AQI, alerts.
- Asset checks for freshness, completeness and value ranges; a sensor that triggers backfills when late data lands.
- Retry policies and resource abstractions (local vs cloud storage) configured via Dagster resources.
- A daily AQI report and alert list materialised to results/, plus instructions to run `dagster dev`.

## Acceptance criteria

- `dagster asset materialize` for a date range works headlessly in tests
- Failed asset checks block downstream alerts
- Backfill of 30 partitions completes
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Sensor drift detection asset
- Comparison write-up vs Airflow DAG design

## Skills this project demonstrates

Data orchestration, Pipeline reliability, Observability

## Definition of done

1. Folder `day-10-air-quality-dagster-assets/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 10 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
