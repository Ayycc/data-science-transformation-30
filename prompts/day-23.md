# Day 23: Observability Analytics: Multivariate Anomaly Detection and Root Cause

**Tier III - Principal** | **Domain:** Technology operations | **Folder:** `day-23-sre-anomaly-root-cause/`

**Core tools:** ruptures, PyOD, statsmodels, causal-learn, NetworkX

**Method:** Change-point detection, multivariate anomaly scoring, causal discovery over service metrics for root-cause ranking.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-23-sre-anomaly-root-cause/`. Do not modify other days' folders.

## Business context

An SRE team for a microservices platform drowns in alerts. During incidents they need to know which service started it, not just that latency is high.

## Data

Simulate metrics (latency, error rate, CPU, queue depth) for a 20-service dependency graph with injected incidents that propagate along the call graph, plus noise and deploy events.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Change-point detection (PELT/binary segmentation) and multivariate anomaly detection (Isolation Forest, ECOD) per service.
- Alert de-duplication into incidents using temporal and topological grouping.
- Causal discovery (PC algorithm / Granger) over metrics plus known topology to rank root-cause candidates.
- Evaluation on injected incidents: detection delay, precision, and root-cause top-1/top-3 accuracy.

## Acceptance criteria

- Root-cause top-3 accuracy reported on at least 30 injected incidents
- Alert volume reduction from grouping quantified
- Deploy events used as a prior and its value measured
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Online version with streaming change-points
- Incident timeline auto-summary

## Skills this project demonstrates

AIOps, Causal discovery, Anomaly detection

## Definition of done

1. Folder `day-23-sre-anomaly-root-cause/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 23 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
