# Day 14: Streaming Predictive Maintenance for Manufacturing IoT

**Tier II - Expert** | **Domain:** Manufacturing | **Folder:** `day-14-iot-streaming-anomaly-detection/`

**Core tools:** River, Bytewax or asyncio, Kafka (optional), DuckDB, Plotly

**Method:** Event-time stream processing, windowed aggregations, online learning and concept-drift detection.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-14-iot-streaming-anomaly-detection/`. Do not modify other days' folders.

## Business context

A factory streams vibration, temperature and current readings from 500 machines. Batch models retrained monthly miss failures and drift with seasonal conditions.

## Data

Simulate a high-rate sensor stream with degradation trajectories leading to failures, out-of-order and late events, sensor dropouts and gradual drift. Optionally replay the NASA C-MAPSS turbofan data.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- A stream processor (Bytewax dataflow or a clean asyncio implementation) with event-time tumbling and sliding windows and watermarks for late data.
- Online anomaly detection (Half-Space Trees) and online remaining-useful-life regression with River, updated per event.
- Drift detectors (ADWIN/Page-Hinkley) triggering model resets; metrics logged to DuckDB.
- Evaluation: precision/recall of failure warnings, lead time before failure, and throughput (events/sec).

## Acceptance criteria

- Late events within the allowed lateness are correctly assigned to windows (tests)
- Median warning lead time is reported and positive
- Throughput of at least 10k events/sec on a laptop-class CPU
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- docker-compose with Redpanda/Kafka
- Exactly-once sink semantics discussion and demo

## Skills this project demonstrates

Stream processing, Online ML, Drift detection

## Definition of done

1. Folder `day-14-iot-streaming-anomaly-detection/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 14 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
