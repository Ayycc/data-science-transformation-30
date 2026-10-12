# Day 22: Crop Yield and Climate Risk with xarray + Dask

**Tier III - Principal** | **Domain:** Agriculture & climate | **Folder:** `day-22-agri-climate-xarray-dask/`

**Core tools:** xarray, Dask, rioxarray, scikit-learn, Zarr

**Method:** Labelled multi-dimensional arrays, chunked out-of-core computation, raster-vector zonal statistics, spatio-temporal ML.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-22-agri-climate-xarray-dask/`. Do not modify other days' folders.

## Business context

An agricultural lender needs field-level yield expectations and drought risk to price loans, using satellite vegetation indices and climate grids.

## Data

Use a small public climate/NDVI sample (for example ERA5 or MODIS via open endpoints) if reachable; otherwise synthesise gridded daily temperature, precipitation and NDVI cubes (time x lat x lon) over a region with drought years, plus field polygons and yields.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Zarr-backed data cube with sensible chunking; Dask-parallel computation of growing degree days, SPI drought index and seasonal NDVI integrals.
- Zonal statistics per field polygon and a yield model with spatial cross-validation (blocked folds).
- Climate-risk scenario: yield distribution under a +2C / -20% precipitation perturbation.
- Performance notes: chunk sizing, task graph size and memory profile.

## Acceptance criteria

- Pipeline processes a cube larger than available RAM via chunking (demonstrated)
- Spatial CV is used and compared to random CV
- Scenario output quantifies yield-at-risk
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- Gaussian process downscaling
- Interactive map of drought exposure

## Skills this project demonstrates

Scientific computing, Geospatial rasters, Climate risk

## Definition of done

1. Folder `day-22-agri-climate-xarray-dask/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 22 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
