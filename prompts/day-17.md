# Day 17: Point-in-Time Correct Feature Store for Ride-Hailing ETA

**Tier III - Principal** | **Domain:** Mobility marketplace | **Folder:** `day-17-ridehail-feature-store-pit/`

**Core tools:** Feast, DuckDB, Redis or SQLite online store, LightGBM, pandas

**Method:** Offline/online feature store, point-in-time joins, training-serving skew prevention.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-17-ridehail-feature-store-pit/`. Do not modify other days' folders.

## Business context

A ride-hailing company's ETA model looked great offline and failed in production because training features leaked future information and online features were computed differently.

## Data

Use NYC TLC trip records if reachable; otherwise simulate trips, driver locations and zone-level traffic with realistic timing.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Feast repository with entities (driver, zone), feature views (rolling traffic, driver acceptance, zone demand) and on-demand features.
- Demonstration of leakage: naive join vs point-in-time join, with the offline/online metric gap quantified.
- Materialisation to an online store and a low-latency retrieval benchmark (p50/p99).
- Skew test: features computed for serving equal those used in training for sampled requests.

## Acceptance criteria

- Point-in-time correctness unit test passes
- Naive vs PIT gap is reported
- Online retrieval p99 under 20 ms locally
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Feature freshness SLAs and monitoring
- Streaming feature push

## Skills this project demonstrates

Feature stores, ML system design, Leakage prevention

## Definition of done

1. Folder `day-17-ridehail-feature-store-pit/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 17 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
