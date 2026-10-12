# Day 02: Card Fraud Detection at Speed with Polars + LightGBM

**Tier I - Advanced** | **Domain:** Financial services | **Folder:** `day-02-card-fraud-polars-lightgbm/`

**Core tools:** Polars, LightGBM, SHAP, scikit-learn, Optuna

**Method:** Lazy-evaluated feature engineering, time-aware validation, cost-sensitive thresholding, explainability.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-02-card-fraud-polars-lightgbm/`. Do not modify other days' folders.

## Business context

A card issuer loses money both to fraud and to false declines that anger good customers. Build a detection model whose operating point is chosen by expected cost, not accuracy.

## Data

Synthetic transaction stream (~5M transactions, 100k cardholders, 6 months) with merchant categories, geolocation, device fingerprints and ~0.3% fraud injected via realistic attack patterns (card testing bursts, account takeover, geo-impossible travel). Use a seeded generator; document every pattern.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Polars LazyFrame pipeline computing rolling velocity, recency, merchant-risk target encoding (leak-free, out-of-fold), and geo-velocity features; benchmark against an equivalent pandas implementation.
- Time-based train/validation/test split with no leakage; LightGBM tuned with Optuna.
- PR-AUC, recall at fixed false-positive rate, and an expected-cost curve that selects the decision threshold from a configurable cost matrix.
- SHAP global and local explanations; reason codes for the top 3 drivers of each alert.

## Acceptance criteria

- Polars pipeline is at least 3x faster than the pandas baseline (report timings)
- No feature uses future information (unit test proves it)
- Threshold choice is justified in dollars
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Model calibration with isotonic regression
- Champion/challenger comparison with a logistic regression baseline

## Skills this project demonstrates

Polars, Gradient boosting, Imbalanced learning, Explainability

## Definition of done

1. Folder `day-02-card-fraud-polars-lightgbm/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 02 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
