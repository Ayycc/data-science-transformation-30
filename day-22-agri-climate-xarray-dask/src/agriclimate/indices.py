"""Lazy (Dask-backed) agro-climate indices on the cube: GDD, heat days, seasonal precipitation, SPI, NDVI integrals.

All functions take and return xarray objects and stay lazy until the caller computes, so the same code runs
on an in-memory test cube or on a chunked Zarr cube larger than memory.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.special import gammainc, ndtri

from .config import GDD_BASE_C, GDD_CAP_C, HEAT_THRESHOLD_C


def open_cube(path: str, chunks: dict | str | None = "stored") -> xr.Dataset:
    """Open the Zarr cube lazily. `chunks="stored"` uses the on-disk chunking as Dask chunks."""
    ds = xr.open_zarr(path, chunks={} if chunks == "stored" else chunks, consolidated=True)
    return ds


def season_weights(time: xr.DataArray, months: tuple[int, int], time_chunks=None) -> xr.DataArray:
    """(time, year) 0/1 matrix: 1 where the day falls in `months` (inclusive range) of that year."""
    years = np.unique(time.dt.year.values)
    m = time.dt.month.values
    in_season = (m >= months[0]) & (m <= months[1])
    w = ((time.dt.year.values[:, None] == years[None, :]) & in_season[:, None]).astype("float32")
    out = xr.DataArray(w, dims=("time", "year"), coords={"time": time.values, "year": years})
    return out.chunk({"time": time_chunks}) if time_chunks is not None else out


def seasonal_sum(da: xr.DataArray, months: tuple[int, int]) -> xr.DataArray:
    """Per-year sum over `months`, as a contraction with a (time, year) indicator matrix.

    Each time chunk contributes a partial (year, lat, lon) sum that Dask tree-reduces, so memory stays at a few
    chunks per worker. `resample(time="YS").sum()` gives identical numbers but in testing held ~4x the memory,
    because each chunk is split at year boundaries into many small tasks (see results/performance.md).
    """
    chunks = da.chunks[da.get_axis_num("time")] if da.chunks is not None else None
    w = season_weights(da["time"], months, chunks)
    return xr.dot(da, w, dim="time").transpose("year", ...)


def seasonal_sum_resample(da: xr.DataArray, months: tuple[int, int]) -> xr.DataArray:
    """Reference implementation with masking + resample (used only for the memory comparison and tests)."""
    month = da["time"].dt.month
    out = da.where((month >= months[0]) & (month <= months[1]), 0.0).resample(time="YS").sum()
    return out.assign_coords(time=out["time"].dt.year).rename(time="year")


def daily_gdd(tmax: xr.DataArray, tmin: xr.DataArray) -> xr.DataArray:
    """Modified growing degree days (base 10 C, tmax capped at 30 C, tmin floored at 10 C)."""
    tx = tmax.clip(max=GDD_CAP_C).clip(min=GDD_BASE_C)
    tn = tmin.clip(min=GDD_BASE_C).clip(max=GDD_CAP_C)
    return ((tx + tn) / 2 - GDD_BASE_C).clip(min=0)


def growing_season_gdd(ds: xr.Dataset) -> xr.DataArray:
    """GDD summed April-September per year."""
    return seasonal_sum(daily_gdd(ds["tmax"], ds["tmin"]), (4, 9)).rename("gdd")


def heat_days(ds: xr.Dataset) -> xr.DataArray:
    """Days with tmax above the heat threshold in June-August."""
    return seasonal_sum((ds["tmax"] > HEAT_THRESHOLD_C).astype("float32"), (6, 8)).rename("heat_days")


def jja_precip(ds: xr.Dataset) -> xr.DataArray:
    """June-August precipitation total (mm): the 3-month accumulation behind SPI-3 for August."""
    return seasonal_sum(ds["precip"], (6, 8)).rename("precip_jja")


def annual_precip(ds: xr.Dataset) -> xr.DataArray:
    return seasonal_sum(ds["precip"], (1, 12)).rename("precip_annual")


# ---------- SPI (McKee et al. 1993) with a two-parameter gamma fitted per pixel ----------


def fit_gamma(x: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thom (1958) maximum-likelihood approximation for gamma shape/scale, with a probability of zero.

    Returns (alpha, beta, q0) reduced over `axis`.
    """
    x = np.moveaxis(np.asarray(x, dtype="float64"), axis, -1)
    pos = x > 0
    q0 = 1.0 - pos.mean(axis=-1)
    xp = np.where(pos, x, np.nan)
    mean = np.nanmean(xp, axis=-1)
    a_stat = np.log(mean) - np.nanmean(np.log(xp), axis=-1)
    a_stat = np.maximum(a_stat, 1e-6)
    alpha = (1.0 + np.sqrt(1.0 + 4.0 * a_stat / 3.0)) / (4.0 * a_stat)
    beta = mean / alpha
    return alpha, beta, q0


def spi_transform(x: np.ndarray, alpha: np.ndarray, beta: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """Map accumulations to standard-normal SPI using already-fitted gamma parameters (broadcasting)."""
    x = np.asarray(x, dtype="float64")
    cdf = q0 + (1.0 - q0) * gammainc(alpha, np.maximum(x, 0) / beta)
    cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
    return ndtri(cdf)


def gamma_params(accum: xr.DataArray, dim: str = "year") -> xr.Dataset:
    """Fit gamma parameters per pixel over `dim` (the dim is rechunked to one chunk; space stays chunked)."""
    a = accum.chunk({dim: -1}) if accum.chunks is not None else accum
    alpha, beta, q0 = xr.apply_ufunc(
        fit_gamma,
        a,
        input_core_dims=[[dim]],
        output_core_dims=[[], [], []],
        dask="parallelized",
        output_dtypes=["float64"] * 3,
    )
    return xr.Dataset({"alpha": alpha, "beta": beta, "q0": q0})


def spi(accum: xr.DataArray, params: xr.Dataset) -> xr.DataArray:
    """SPI of `accum` relative to the (baseline) gamma `params`. Pass baseline params to score scenarios."""
    out = xr.apply_ufunc(
        spi_transform,
        accum,
        params["alpha"],
        params["beta"],
        params["q0"],
        dask="parallelized",
        output_dtypes=["float64"],
    )
    return out.rename("spi3_aug")


def ndvi_season(ds: xr.Dataset) -> xr.Dataset:
    """Seasonal NDVI integral (mean May-Sep NDVI x 153 days, 'NDVI-days') and seasonal peak, per year."""
    nd = ds["ndvi"]
    month = nd["time_ndvi"].dt.month
    season = nd.where((month >= 5) & (month <= 9))
    grp = season.groupby(nd["time_ndvi"].dt.year)
    integral = (grp.mean("time_ndvi", skipna=True) * 153.0).rename("ndvi_int")
    peak = grp.max("time_ndvi", skipna=True).rename("ndvi_peak")
    return xr.merge([integral, peak])


def climate_features(ds: xr.Dataset, params: xr.Dataset | None = None) -> tuple[xr.Dataset, xr.Dataset]:
    """All annual climate features (lazy). If `params` is None, gamma parameters are fitted on `ds`."""
    p_jja = jja_precip(ds)
    if params is None:
        params = gamma_params(p_jja)
    feats = xr.merge([growing_season_gdd(ds), heat_days(ds), p_jja, spi(p_jja, params)])
    return feats, params


def perturb(ds: xr.Dataset, dt_c: float, precip_factor: float) -> xr.Dataset:
    """Delta-change climate scenario (lazy): shift temperatures by dt_c and scale precipitation."""
    out = ds.copy()
    out["tmax"] = ds["tmax"] + dt_c
    out["tmin"] = ds["tmin"] + dt_c
    out["precip"] = ds["precip"] * precip_factor
    return out


def compute_climate_features(ds: xr.Dataset, params: xr.Dataset | None = None, num_workers: int = 4) -> tuple[xr.Dataset, xr.Dataset, dict]:
    """Compute the annual features one reduction at a time (bounded memory), then SPI on the small result.

    Computing each reduction separately keeps only one feature's working set of chunks in memory; a single
    merged graph over all features held about twice the memory in testing. Returns (features, params, tasks).
    """
    import dask

    lazy = {"gdd": growing_season_gdd(ds), "heat_days": heat_days(ds), "precip_jja": jja_precip(ds)}
    tasks = {k: int(len(v.data.__dask_graph__())) for k, v in lazy.items() if v.chunks is not None}
    with dask.config.set(scheduler="threads", num_workers=num_workers):
        done = {k: v.compute() for k, v in lazy.items()}
    if params is None:
        params = gamma_params(done["precip_jja"])
    feats = xr.Dataset(done)
    feats["spi3_aug"] = spi(done["precip_jja"], params)
    return feats, params, tasks
