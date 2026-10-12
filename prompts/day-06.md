# Day 06: Experimentation Engine: CUPED, Sequential and Bayesian Tests

**Tier I - Advanced** | **Domain:** Digital product & marketing | **Folder:** `day-06-experimentation-platform-cuped/`

**Core tools:** SciPy, statsmodels, NumPy, pandas, Streamlit

**Method:** Variance reduction (CUPED), sample-ratio mismatch checks, always-valid sequential tests, Bayesian decision rules.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-06-experimentation-platform-cuped/`. Do not modify other days' folders.

## Business context

A subscription app runs dozens of A/B tests a month. Analysts peek at results daily, inflating false positives, and small effects need impractically long tests.

## Data

Simulate users with pre-period engagement, multiple metrics (conversion, revenue, retention) and experiments with known true effects, novelty effects and a deliberate SRM bug.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- A small Python library: power/sample-size calculator, SRM detector, CUPED estimator, delta-method ratio metrics, mSPRT always-valid p-values, and a Beta-Binomial / Normal Bayesian analysis.
- A Monte Carlo study proving false-positive control under daily peeking and the variance reduction achieved by CUPED.
- An experiment readout generator producing a one-page Markdown report per test.
- Optional Streamlit app for interactive analysis.

## Acceptance criteria

- Simulated FPR under peeking stays at or below 5% for the sequential method
- CUPED reduces variance by at least 30% on correlated metrics
- Library has unit tests with at least 85% coverage
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Heterogeneous treatment effect slices with multiple-testing correction
- Interleaving for ranking experiments

## Skills this project demonstrates

Statistical inference, Experiment design, Library design

## Definition of done

1. Folder `day-06-experimentation-platform-cuped/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 06 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
