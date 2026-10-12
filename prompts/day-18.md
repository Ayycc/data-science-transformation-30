# Day 18: Bayesian Marketing Mix Model and Budget Optimiser

**Tier III - Principal** | **Domain:** Consumer goods & marketing | **Folder:** `day-18-bayesian-marketing-mix-model/`

**Core tools:** PyMC, PyMC-Marketing, ArviZ, SciPy, pandas

**Method:** Hierarchical Bayesian regression with adstock and saturation, posterior predictive checks, budget optimisation under uncertainty.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-18-bayesian-marketing-mix-model/`. Do not modify other days' folders.

## Business context

A consumer brand spends heavily across TV, search, social and retail media. Last-click attribution over-credits search and the CFO wants a defensible budget split.

## Data

Simulate 3 years of weekly sales across 10 regions with known channel effects, carryover, diminishing returns, seasonality, price and competitor activity, so recovered parameters can be validated.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Bayesian MMM with geometric adstock and Hill/logistic saturation, hierarchical across regions.
- Prior predictive and posterior predictive checks, convergence diagnostics (R-hat, ESS), and parameter recovery vs truth.
- Channel contribution and ROAS posteriors with credible intervals.
- Budget optimiser that maximises expected sales (and a risk-adjusted version) under total-budget and channel bounds.

## Acceptance criteria

- All R-hat below 1.01
- True channel ROAS lies within the 94% HDI for most channels
- Optimised budget shows expected lift with uncertainty bands
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Calibration with lift-test priors
- Time-varying coefficients via Gaussian random walk

## Skills this project demonstrates

Bayesian modelling, Marketing science, Decision under uncertainty

## Definition of done

1. Folder `day-18-bayesian-marketing-mix-model/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 18 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
