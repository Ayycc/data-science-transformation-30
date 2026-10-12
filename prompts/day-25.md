# Day 25: Privacy-Preserving Analytics: Differential Privacy and Synthetic Data

**Tier IV - Distinguished** | **Domain:** Census & health statistics | **Folder:** `day-25-privacy-dp-synthetic-data/`

**Core tools:** OpenDP or diffprivlib, SDV, scikit-learn, pandas, Anonymeter

**Method:** Differentially private releases, privacy budget accounting, synthetic data generation with utility and disclosure-risk evaluation.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-25-privacy-dp-synthetic-data/`. Do not modify other days' folders.

## Business context

A statistics office must share microdata and aggregate tables with researchers while provably protecting individuals.

## Data

Use the public UCI Adult (census income) dataset if reachable; otherwise generate a census-like population with realistic correlations and rare subgroups.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- DP aggregate release (counts, means, histograms) with Laplace/Gaussian mechanisms and a tracked privacy budget; accuracy vs epsilon curves.
- DP model training (diffprivlib logistic regression or DP-SGD) vs non-private baseline.
- Synthetic datasets from Gaussian copula, CTGAN and a DP synthesiser; utility (train-synthetic-test-real, marginal fidelity).
- Disclosure-risk assessment: k-anonymity, membership inference and singling-out/linkability attacks.
- A privacy-utility frontier chart and a release recommendation memo.

## Acceptance criteria

- Privacy budget never exceeds the configured total (test)
- Utility and risk reported for each synthesiser
- Rare-subgroup accuracy loss explicitly analysed
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Zero-concentrated DP accounting
- Query-answering server with budget enforcement

## Skills this project demonstrates

Differential privacy, Synthetic data, Statistical disclosure control

## Definition of done

1. Folder `day-25-privacy-dp-synthetic-data/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 25 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
