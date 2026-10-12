# Day 24: Recommender System with Two-Tower Retrieval and Off-Policy Evaluation

**Tier IV - Distinguished** | **Domain:** Media streaming | **Folder:** `day-24-streaming-recsys-off-policy-eval/`

**Core tools:** PyTorch, FAISS, implicit, NumPy, pandas

**Method:** Two-stage recommendation (retrieval + ranking), implicit feedback, counterfactual evaluation with IPS/SNIPS/doubly robust estimators.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-24-streaming-recsys-off-policy-eval/`. Do not modify other days' folders.

## Business context

A streaming service wants to replace its popularity-based homepage. Online A/B slots are scarce, so new policies must be screened offline without the bias of logged data.

## Data

Use MovieLens (1M or 25M sample) if reachable; additionally simulate a logging policy with known propensities over a synthetic user-item world so off-policy estimates can be checked against ground truth.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Baselines: popularity, item-kNN, ALS (implicit).
- Two-tower retrieval model in PyTorch with in-batch negatives, served via a FAISS ANN index; a gradient-boosted re-ranker.
- Offline metrics: recall@k, NDCG@k, coverage, novelty, and popularity-bias analysis.
- Off-policy evaluation of candidate policies with IPS, SNIPS and DR estimators; bias/variance vs true simulated online reward.

## Acceptance criteria

- Two-tower + re-ranker beats ALS on NDCG@10
- DR estimate is closest to the simulated truth among estimators (or explain why not)
- Retrieval latency p99 reported
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Diversity-aware re-ranking (MMR)
- Cold-start with content features

## Skills this project demonstrates

Recommender systems, Deep learning, Counterfactual evaluation

## Definition of done

1. Folder `day-24-streaming-recsys-off-policy-eval/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 24 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
