# Day 12: Did the Programme Work? Causal Inference with Double ML

**Tier II - Expert** | **Domain:** Public policy & workforce | **Folder:** `day-12-policy-causal-inference-dml/`

**Core tools:** DoWhy, EconML, statsmodels, linearmodels, NetworkX

**Method:** Causal DAGs, identification, propensity methods, difference-in-differences, double machine learning, refutation tests.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-12-policy-causal-inference-dml/`. Do not modify other days' folders.

## Business context

A state launched a job-training programme in some counties. Leadership wants to know its effect on employment and earnings, and for whom it works best, before scaling it.

## Data

Simulate a county-by-quarter panel and individual-level records with confounded enrolment, staggered rollout and known heterogeneous effects so estimates can be validated against truth. Optionally replicate on the public LaLonde/NSW data.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Explicit causal DAG, adjustment-set reasoning with DoWhy, and a naive vs adjusted comparison.
- Propensity score matching/IPW with balance diagnostics.
- Staggered difference-in-differences (Callaway-Sant'Anna or event-study style) with parallel-trends checks.
- Double ML / causal forest CATE estimates, with refutation tests (placebo treatment, random common cause, subset).
- A policy memo (Markdown) that states the effect, uncertainty and recommendation in plain language.

## Acceptance criteria

- Estimated ATE within the 95% CI of the known true effect
- Naive estimate's bias is quantified
- At least three refutation tests reported
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Synthetic control for a single treated state
- Sensitivity analysis to unobserved confounding (E-values)

## Skills this project demonstrates

Causal inference, Econometrics, Policy communication

## Definition of done

1. Folder `day-12-policy-causal-inference-dml/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 12 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
