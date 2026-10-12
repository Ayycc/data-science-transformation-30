# Day 16: Insurance Claims Severity: Full MLOps Lifecycle

**Tier III - Principal** | **Domain:** Insurance | **Folder:** `day-16-insurance-claims-mlops/`

**Core tools:** MLflow, FastAPI, Evidently, Docker, GitHub Actions

**Method:** Experiment tracking, model registry, containerised serving, CI/CD and production drift monitoring.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-16-insurance-claims-mlops/`. Do not modify other days' folders.

## Business context

An insurer's pricing team ships models via emailed pickle files. Nobody knows which model is live or whether it still performs.

## Data

Use the public French motor TPL (freMTPL2) dataset if reachable via OpenML; otherwise simulate policies and claims with Tweedie-distributed severity and covariate drift over time.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Training pipeline (GLM Tweedie vs GBM) tracked in MLflow with parameters, metrics, artefacts and a registered model with stages.
- FastAPI scoring service with Pydantic request validation, health/readiness endpoints, batch and single scoring, and a Dockerfile.
- GitHub Actions workflow for this folder: lint, tests, model validation gate (must beat current champion), image build.
- Evidently drift and performance reports over simulated monthly production batches, with a retraining trigger policy.

## Acceptance criteria

- API contract tests pass with TestClient
- Model promotion is automatic only if the gate passes
- Drift report flags the injected covariate shift
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Shadow deployment and canary comparison
- Model card generated from MLflow metadata

## Skills this project demonstrates

MLOps, API engineering, Monitoring

## Definition of done

1. Folder `day-16-insurance-claims-mlops/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 16 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
