# Day 13: Anti-Money-Laundering with Transaction Graph Analytics

**Tier II - Expert** | **Domain:** Banking & compliance | **Folder:** `day-13-aml-transaction-graph-analytics/`

**Core tools:** NetworkX, igraph, PyTorch Geometric (optional), scikit-learn, pandas

**Method:** Graph construction, community detection, motif mining, graph features and node classification.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-13-aml-transaction-graph-analytics/`. Do not modify other days' folders.

## Business context

A bank's rules-based AML system generates thousands of false alerts while sophisticated layering schemes go unnoticed because they only show up in the network.

## Data

Use the public IBM AML synthetic dataset if reachable; otherwise generate a transaction graph (~100k accounts, ~2M edges) with injected typologies: fan-in/fan-out, cycles, scatter-gather, mule chains.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Temporal graph construction and summary statistics.
- Community detection (Louvain/Leiden) and cycle/motif detection for known typologies.
- Node features (PageRank, betweenness, flow ratios, temporal burstiness) feeding a classifier; optional GNN (GraphSAGE) comparison.
- Investigator view: subgraph visualisations for top alerts exported as HTML (pyvis).

## Acceptance criteria

- Each injected typology is detected at a reported recall
- Graph features beat tabular-only features on PR-AUC
- Alerts include an explainable subgraph
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Temporal motif counting
- Graph embedding with node2vec

## Skills this project demonstrates

Graph analytics, Financial crime, Explainable alerts

## Definition of done

1. Folder `day-13-aml-transaction-graph-analytics/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 13 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
