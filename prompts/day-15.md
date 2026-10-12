# Day 15: Bike-Share Demand and Fleet Rebalancing

**Tier II - Expert** | **Domain:** Urban mobility & transport | **Folder:** `day-15-bikeshare-geospatial-rebalancing/`

**Core tools:** GeoPandas, H3, OSMnx, LightGBM, OR-Tools

**Method:** Spatial feature engineering, spatio-temporal demand prediction and vehicle-routing optimisation.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-15-bikeshare-geospatial-rebalancing/`. Do not modify other days' folders.

## Business context

A city bike-share operator has empty docks downtown every morning and overflowing docks every evening. Rebalancing trucks run fixed routes.

## Data

Use public trip data (Citi Bike, Divvy or Capital Bikeshare) if reachable; otherwise synthesise trips over a real city's station layout with commute patterns and weather.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Station and H3-cell features (POI density from OSM if reachable, transit proximity, elevation proxy).
- Hourly station-level net-flow forecasts with spatial lags.
- Capacitated vehicle routing (OR-Tools) for overnight and midday rebalancing given forecast imbalance.
- Before/after simulation of empty/full dock minutes, with interactive maps (folium/kepler) saved to results/.

## Acceptance criteria

- Forecast beats historical-average baseline on station-level MAE
- Routing respects truck capacity and shift length
- Simulated stockout minutes reduced by a reported percentage
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Dynamic incentive pricing for user-led rebalancing
- Equity of access analysis

## Skills this project demonstrates

Geospatial data science, Spatio-temporal ML, Vehicle routing

## Definition of done

1. Folder `day-15-bikeshare-geospatial-rebalancing/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 15 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
