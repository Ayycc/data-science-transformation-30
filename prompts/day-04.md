# Day 04: Grid Load Forecasting with Probabilistic Backtesting

**Tier I - Advanced** | **Domain:** Energy & utilities | **Folder:** `day-04-energy-load-forecasting-conformal/`

**Core tools:** statsforecast, LightGBM, MAPIE, pandas, Plotly

**Method:** Classical vs ML forecasting, rolling-origin backtests, conformal prediction intervals.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-04-energy-load-forecasting-conformal/`. Do not modify other days' folders.

## Business context

A grid operator schedules generation a day ahead. Under-forecasting causes expensive spot purchases; over-forecasting wastes reserve. They need calibrated intervals, not just point forecasts.

## Data

Prefer public hourly load data (for example EIA-930 or ENTSO-E if reachable without keys); otherwise generate 4 years of hourly load for 5 regions with daily/weekly/annual seasonality, temperature-driven demand, holidays and a regime shift.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Baselines: seasonal naive, ETS/AutoARIMA and MSTL via statsforecast.
- Global LightGBM model with lag, calendar, holiday and weather features.
- Rolling-origin backtest over 12 folds reporting MASE, sMAPE and pinball loss.
- Conformal prediction intervals with empirical coverage per region and horizon.

## Acceptance criteria

- 90% intervals achieve 87-93% empirical coverage in backtest
- Best model beats seasonal naive MASE by at least 15%
- Backtest is a single reproducible command
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Hierarchical reconciliation across regions
- Temperature scenario stress test

## Skills this project demonstrates

Time-series forecasting, Backtesting, Uncertainty quantification

## Definition of done

1. Folder `day-04-energy-load-forecasting-conformal/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 04 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
