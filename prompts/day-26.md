# Day 26: Responsible Credit Decisioning: Fairness, Explainability and Model Governance

**Tier IV - Distinguished** | **Domain:** Lending | **Folder:** `day-26-lending-fairness-model-governance/`

**Core tools:** Fairlearn, SHAP, scikit-learn, LightGBM, Quarto or Markdown

**Method:** Fairness metrics and mitigation, counterfactual explanations, adverse-action reason codes, model risk documentation (SR 11-7 style).

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-26-lending-fairness-model-governance/`. Do not modify other days' folders.

## Business context

A lender is modernising its credit model. Regulators require fair-lending analysis, adverse-action explanations and documented model risk management before deployment.

## Data

Use the public HMDA mortgage data sample or the German Credit dataset if reachable; otherwise generate applications with proxy variables for protected attributes and historical bias baked into labels.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Baseline scorecard (WOE logistic regression) vs gradient boosting; performance and stability (PSI).
- Fairness audit: demographic parity, equalised odds, and adverse impact ratio with bootstrapped CIs, including proxy discrimination analysis.
- Mitigation: reweighing, threshold optimisation and exponentiated gradient, with the accuracy-fairness trade-off frontier.
- Adverse-action reason codes from SHAP, and counterfactual 'what would change the decision' explanations.
- Model development document and model card covering limitations, monitoring and validation evidence.

## Acceptance criteria

- Adverse impact ratio reported before and after mitigation
- Every declined applicant gets 4 ranked reason codes
- Governance document covers data, methodology, testing, limitations and monitoring
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Less-discriminatory-alternative search
- Monotonic constraints for regulatory interpretability

## Skills this project demonstrates

Responsible AI, Model risk management, Credit risk

## Definition of done

1. Folder `day-26-lending-fairness-model-governance/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 26 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
