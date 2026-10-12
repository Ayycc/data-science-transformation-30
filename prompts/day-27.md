# Day 27: Dynamic Pricing for Hospitality with Contextual Bandits

**Tier IV - Distinguished** | **Domain:** Travel & hospitality | **Folder:** `day-27-hospitality-dynamic-pricing-bandits/`

**Core tools:** NumPy, PyTorch (optional), SciPy, pandas, Plotly

**Method:** Demand-curve estimation, contextual bandits (LinUCB, Thompson sampling), constrained revenue optimisation and safe exploration.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-27-hospitality-dynamic-pricing-bandits/`. Do not modify other days' folders.

## Business context

A hotel group sets room rates with static seasonal tables and leaves money on the table during demand spikes, while aggressive pricing risks brand damage.

## Data

Build a market simulator: guests with willingness-to-pay that depends on lead time, season, events, competitor prices and segment; inventory that perishes nightly.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Price-elasticity estimation from logged historical data with confounding addressed (instrument or randomised price tests).
- Policies: static table, greedy elasticity-based optimiser, epsilon-greedy, LinUCB and Thompson sampling with price grids.
- Inventory-aware revenue management (bid-price or dynamic-programming control) combined with the bandit.
- Simulation study over many seeds: revenue, RevPAR, regret, price volatility, and guardrail violations.

## Acceptance criteria

- Bandit policies achieve sublinear regret curves in simulation
- Guardrails (max daily price change, floor/ceiling) are never violated (test)
- Revenue uplift vs static table reported with confidence intervals
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Off-policy evaluation of the bandit from logs
- Competitor reaction modelling

## Skills this project demonstrates

Sequential decision-making, Pricing science, Simulation

## Definition of done

1. Folder `day-27-hospitality-dynamic-pricing-bandits/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 27 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
