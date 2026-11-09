# Day 22: Crop Yield and Climate Risk with xarray + Dask

**Tier III - Principal** | **Domain:** Agriculture & climate | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-22-agri-climate-xarray-dask/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-22-agri-climate-xarray-dask/README.md)

- **Key metric:** +2C/-20% rain mean yield change (model): **-22.1%** (truth -23.8%)
- **Data:** synthetic fallback (public source blocked); public sources attempted: Open-Meteo ERA5; reachable: none
- **Tests (static):** 25 functions, 25 cases (25 stated in README)
- **Lines of code:** 1,391 Python source, 285 tests, 0 SQL
- **CI:** success
- **Tools:** dask, matplotlib, numcodecs, numpy, pandas, pyproj, rasterio, rioxarray, scikit-learn, scipy, shapely, xarray, zarr

## Caveats

- Gaussian-process downscaling (stretch) not attempted
