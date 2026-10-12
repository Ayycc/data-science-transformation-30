# Day 30: Capstone: Decision-Intelligence Digital Twin and Portfolio Retrospective

**Tier IV - Distinguished** | **Domain:** Cross-industry (retail supply chain) | **Folder:** `day-30-capstone-supply-chain-digital-twin/`

**Core tools:** SimPy, OR-Tools, PyMC, Streamlit, Quarto or MkDocs

**Method:** Integrates forecasting, causal effects, optimisation and simulation into one what-if decision system, then audits the entire 30-day portfolio.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-30-capstone-supply-chain-digital-twin/`. Do not modify other days' folders.

## Business context

A retailer's leadership wants to test strategic decisions (new DC, promo strategy, supplier diversification) before committing capital. This capstone also closes the 30-day challenge.

## Data

Reuse and extend synthetic generators from earlier days (retail lakehouse, inventory optimisation, MMM) or re-create compatible ones inside this folder so it runs standalone.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- A digital twin: SimPy network simulation (suppliers, DCs, stores) driven by probabilistic demand and promo-effect models, with disruption scenarios.
- Scenario engine evaluating strategic options with risk metrics (expected profit, CVaR, service level) and an optimiser over decisions.
- Streamlit app for what-if exploration.
- Portfolio retrospective: a script that scans all day-* folders and produces PORTFOLIO_REPORT.md (tools coverage matrix, test results, lines of code, key metric per project, lessons learned), plus a docs site (MkDocs or Quarto) indexing all 30 projects.

## Acceptance criteria

- Twin reproduces baseline KPIs within 5% of historical synthetic data
- At least 3 strategic scenarios compared with uncertainty
- PORTFOLIO_REPORT.md covers all 30 days
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Reinforcement-learning replenishment agent inside the twin
- Executive one-pager PDF

## Skills this project demonstrates

Decision intelligence, Systems integration, Technical leadership

## Definition of done

1. Folder `day-30-capstone-supply-chain-digital-twin/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 30 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
