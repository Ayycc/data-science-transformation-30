# Day 07: Demand-Driven Inventory Optimisation

**Tier I - Advanced** | **Domain:** Supply chain & logistics | **Folder:** `day-07-supply-chain-demand-inventory-opt/`

**Core tools:** OR-Tools, PuLP, statsforecast, pandas, SimPy

**Method:** Probabilistic demand forecasting feeding stochastic inventory policies and a mixed-integer replenishment model, validated by simulation.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-07-supply-chain-demand-inventory-opt/`. Do not modify other days' folders.

## Business context

A distributor with 3 warehouses and 800 SKUs carries too much slow-moving stock while best-sellers stock out. Planners set reorder points by gut feel.

## Data

Use the M5 (Walmart) sample if reachable; otherwise synthesise intermittent and smooth demand series with lead-time variability, supplier MOQs and holding/shortage costs.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- ABC/XYZ segmentation and quantile demand forecasts (including Croston/TSB for intermittent SKUs).
- Safety-stock and (s, S) policy per SKU from service-level targets; newsvendor for seasonal items.
- MILP replenishment plan across warehouses with capacity, MOQ and budget constraints (OR-Tools or PuLP).
- SimPy discrete-event simulation comparing current heuristic vs optimised policy on fill rate, inventory value and cost.

## Acceptance criteria

- Optimised policy improves total cost by a reported margin at equal or better fill rate in simulation
- MILP solves to optimality or a documented gap
- All costs are parameterised in a config file
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Multi-echelon extension
- Sensitivity analysis on lead-time variance

## Skills this project demonstrates

Operations research, Forecasting, Simulation

## Definition of done

1. Folder `day-07-supply-chain-demand-inventory-opt/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 07 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
