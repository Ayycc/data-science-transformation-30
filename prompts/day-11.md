# Day 11: Support Ticket Intelligence: NLP Classification with Active Learning

**Tier II - Expert** | **Domain:** Customer operations | **Folder:** `day-11-support-ticket-nlp-active-learning/`

**Core tools:** spaCy, sentence-transformers, scikit-learn, modAL or custom, UMAP

**Method:** Text normalisation, embedding-based classification, uncertainty-sampling active learning and topic discovery.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-11-support-ticket-nlp-active-learning/`. Do not modify other days' folders.

## Business context

A SaaS support team receives 10k tickets a week. Routing is manual and slow, and nobody has time to label much data.

## Data

Use a public customer-support corpus (for example a Hugging Face dataset such as bitext customer support) if reachable; otherwise generate templated-plus-noisy tickets across 15 intents with typos, multilingual snippets and PII.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- PII redaction step (emails, phones, names) with spaCy NER and regex, with tests.
- Baselines: TF-IDF + linear model; embedding (small sentence-transformer) + logistic regression.
- Active learning loop simulating an annotator with a 2% label budget; learning curves vs random sampling.
- Unsupervised topic discovery (embeddings + UMAP + HDBSCAN or BERTopic) to surface emerging issues.

## Acceptance criteria

- Active learning reaches the random-sampling F1 with at most 50% of the labels
- PII redaction recall at least 95% on a labelled test set
- Macro-F1 per intent reported with confusion matrix
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Calibrated abstention: route low-confidence tickets to humans
- Weak supervision with labelling functions

## Skills this project demonstrates

NLP, Active learning, Data privacy

## Definition of done

1. Folder `day-11-support-ticket-nlp-active-learning/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 11 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
