# Day 08: Telecom Retention: Survival Analysis + Uplift Modelling

**Tier II - Expert** | **Domain:** Telecommunications | **Folder:** `day-08-telecom-churn-survival-uplift/`

**Core tools:** lifelines, scikit-survival, CausalML, scikit-learn, pandas

**Method:** Time-to-event modelling of churn, then uplift models to target only customers whose behaviour a retention offer changes.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-08-telecom-churn-survival-uplift/`. Do not modify other days' folders.

## Business context

A carrier sends retention discounts to everyone predicted to churn, wasting money on customers who would stay anyway and annoying 'sleeping dogs'.

## Data

Synthetic subscriber panel (~200k customers, 36 months) with contract types, usage, network quality, support tickets, censoring, and a historical randomised retention campaign with heterogeneous true effects.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Kaplan-Meier curves by segment, Cox PH with diagnostics (Schoenfeld residuals), and a random survival forest; concordance and time-dependent AUC.
- Uplift models (T-learner, X-learner, uplift random forest) evaluated with Qini and uplift curves against the known ground truth.
- A targeting policy that maximises incremental retained revenue under a budget, with expected ROI.

## Acceptance criteria

- Proportional hazards assumption is tested and violations handled
- Uplift targeting beats churn-score targeting on incremental revenue in holdout
- Customer lifetime value is computed from the survival curves
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Competing risks (churn vs port-out vs non-payment)
- Policy learning with doubly robust scores

## Skills this project demonstrates

Survival analysis, Uplift modelling, Decision science

## Definition of done

1. Folder `day-08-telecom-churn-survival-uplift/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 08 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
