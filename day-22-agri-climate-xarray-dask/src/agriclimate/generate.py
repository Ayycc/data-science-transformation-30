"""Seeded SYNTHETIC gridded climate + NDVI generator, written tile by tile into a Zarr store.

Every value produced here is synthetic. The generator is built so that a cube of any size can be written
without holding it in memory: each (spatial tile, all years) block is generated, written into its Zarr region
and dropped. Smooth spatial structure comes from a fixed set of random Fourier basis functions evaluated on
the tile's own coordinates, so large-scale weather is coherent across tiles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import ndtri

from .config import DROUGHT_YEARS, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, CubeSpec

NDVI_STEP_DAYS = 8
N_COMPOSITES = 46  # 8-day composites, day of year 1, 9, ..., 361 (MODIS-style)


def grid(spec: CubeSpec) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinates. Latitude descends (north-up raster), longitude ascends."""
    lat = LAT_MAX - spec.res_lat * (np.arange(spec.n_lat) + 0.5)
    lon = LON_MIN + spec.res_lon * (np.arange(spec.n_lon) + 0.5)
    return lat.round(6), lon.round(6)


def time_index(years: tuple[int, ...]) -> pd.DatetimeIndex:
    return pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")


def ndvi_time_index(years: tuple[int, ...]) -> pd.DatetimeIndex:
    out = []
    for y in years:
        start = pd.Timestamp(f"{y}-01-01")
        out.extend(start + pd.to_timedelta(NDVI_STEP_DAYS * np.arange(N_COMPOSITES), unit="D"))
    return pd.DatetimeIndex(out)


def basis_points(la: np.ndarray, lo: np.ndarray, seed: int, k: int, wl_min: float, wl_max: float) -> np.ndarray:
    """k random Fourier features at arbitrary points, shape (k, n_points), roughly unit variance."""
    rng = np.random.default_rng([seed, 7, k])
    theta = rng.uniform(0, 2 * np.pi, k)
    wl = rng.uniform(wl_min, wl_max, k)
    phase = rng.uniform(0, 2 * np.pi, k)
    kx = (2 * np.pi / wl) * np.cos(theta)
    ky = (2 * np.pi / wl) * np.sin(theta)
    arg = kx[:, None] * np.ravel(lo)[None, :] + ky[:, None] * np.ravel(la)[None, :] + phase[:, None]
    return np.sqrt(2.0 / k) * np.cos(arg)


def basis(lat: np.ndarray, lon: np.ndarray, seed: int, k: int, wl_min: float, wl_max: float) -> np.ndarray:
    """Fourier features on the (lat, lon) grid, shape (k, n_lat * n_lon)."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    return basis_points(la, lo, seed, k, wl_min, wl_max)


def soil_index(lat: np.ndarray, lon: np.ndarray, seed: int) -> np.ndarray:
    """Static, spatially smooth soil productivity multiplier (~0.8-1.2). Hidden from the yield model."""
    phi = basis(lat, lon, seed + 2, 12, 1.5, 6.0)
    c = np.random.default_rng([seed, 2]).normal(size=12)
    return np.clip(1.0 + 0.11 * (c @ phi), 0.75, 1.25).reshape(lat.size, lon.size)


def drought_footprint(year: int, lat: np.ndarray, lon: np.ndarray, seed: int) -> np.ndarray:
    """Intensity x Gaussian footprint of a drought for one year (zeros if not a drought year)."""
    out = np.zeros((lat.size, lon.size))
    intensity = DROUGHT_YEARS.get(year, 0.0)
    if intensity == 0.0:
        return out
    rng = np.random.default_rng([seed, year, 99])
    c_lat = rng.uniform(LAT_MIN + 0.8, LAT_MAX - 0.8)
    c_lon = rng.uniform(LON_MIN + 1.0, LON_MAX - 1.0)
    radius = rng.uniform(1.6, 3.0)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    d2 = (la - c_lat) ** 2 + ((lo - c_lon) * np.cos(np.deg2rad(c_lat))) ** 2
    return intensity * np.exp(-d2 / (2 * radius**2))


def _season(doy: np.ndarray, peak: float) -> np.ndarray:
    return np.sin(2 * np.pi * (doy - peak) / 365.25)


def summer_window(doy: np.ndarray) -> np.ndarray:
    """Smooth 0-1 weight covering roughly May to August, when droughts act."""
    return np.exp(-0.5 * ((doy - 185) / 35.0) ** 2)


def expected_daily_precip(doy: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Climatological mean precipitation (mm/day) implied by the generator, shape (n_doy, n_lon)."""
    p_wet = wet_probability(doy[:, None], lon[None, :])
    return p_wet * mean_wet_amount(doy[:, None], lon[None, :])


def wet_probability(doy: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.clip(0.27 + 0.07 * _season(doy, 100) + 0.04 * (lon + 93.0) / 3.0, 0.05, 0.6)


def mean_wet_amount(doy: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return 8.5 + 3.5 * _season(doy, 100) + 1.2 * (lon + 93.0) / 3.0


def generate_year(year: int, lat: np.ndarray, lon: np.ndarray, seed: int, tile_key: tuple[int, int]) -> dict[str, np.ndarray]:
    """Daily tmax, tmin (degC), precip (mm/day) and 8-day NDVI for one year on one tile."""
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    n = len(dates)
    doy = dates.day_of_year.to_numpy().astype(float)
    nlat, nlon = lat.size, lon.size
    npx = nlat * nlon
    la = np.repeat(lat, nlon)
    lo = np.tile(lon, nlat)
    k = 16
    phi_w = basis(lat, lon, seed, k, 3.0, 12.0)  # synoptic weather scale
    phi_y = basis(lat, lon, seed + 1, k, 2.0, 8.0)  # seasonal anomaly scale
    rng_y = np.random.default_rng([seed, year])  # shared by all tiles of this year
    rng_px = np.random.default_rng([seed, year, tile_key[0], tile_key[1]])  # tile-local noise

    w_sum = summer_window(doy)[:, None]
    drought = drought_footprint(year, lat, lon, seed).ravel()[None, :]

    # --- temperature ---
    phi_ar = 0.75
    innov = rng_y.normal(size=(n, k)) * np.sqrt(1 - phi_ar**2)
    a = np.empty_like(innov)
    a[0] = rng_y.normal(size=k)
    for t in range(1, n):
        a[t] = phi_ar * a[t - 1] + innov[t]
    anom_scale = (3.0 + 1.5 * np.cos(2 * np.pi * (doy - 15) / 365.25))[:, None]
    t_anom = anom_scale * (a @ phi_w)
    c_year = rng_y.normal(size=k)
    summer_anom = 0.9 * (c_year @ phi_y)[None, :] * w_sum
    clim = 9.3 - 0.9 * (la - 42.0) + 0.15 * (lo + 93.0)
    seasonal = 14.5 * _season(doy, 110)[:, None]
    tmean = clim[None, :] + seasonal + t_anom + summer_anom + 3.0 * drought * w_sum
    tmean += rng_px.normal(0, 0.6, size=(n, npx))
    dtr = 11.5 + 1.5 * _season(doy, 100)[:, None] + 2.0 * drought * w_sum
    tmax = tmean + dtr / 2
    tmin = tmean - dtr / 2

    # --- precipitation: latent Gaussian occurrence + gamma amounts ---
    b = rng_y.normal(size=(n, k))
    z = 0.8 * (b @ phi_w) + 0.6 * rng_px.normal(size=(n, npx))
    p_wet = wet_probability(doy[:, None], lo[None, :])
    wet = z > ndtri(1 - p_wet)
    shape = 0.75
    mean_amt = mean_wet_amount(doy[:, None], lo[None, :])
    year_factor = np.exp(0.18 * (rng_y.normal(size=k) @ phi_y))[None, :]
    mult = (1.0 - 0.65 * drought * w_sum) * year_factor
    amount = rng_px.gamma(shape, mean_amt / shape, size=(n, npx)) * mult
    precip = np.where(wet, amount, 0.0)

    soil = soil_index(lat, lon, seed).ravel()
    ndvi = ndvi_from_climate(tmax, tmean, precip, doy, lo, soil, rng_px)
    shp = (n, nlat, nlon)
    return {
        "tmax": tmax.reshape(shp).astype("float32"),
        "tmin": tmin.reshape(shp).astype("float32"),
        "precip": precip.reshape(shp).astype("float32"),
        "ndvi": ndvi.reshape((N_COMPOSITES, nlat, nlon)).astype("float32"),
    }


def ndvi_from_climate(
    tmax: np.ndarray,
    tmean: np.ndarray,
    precip: np.ndarray,
    doy: np.ndarray,
    lo: np.ndarray,
    soil: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """8-day NDVI driven by phenology, soil, trailing water supply and heat (works for any daily climate).

    Arrays are (time, pixel); `lo` and `soil` are per pixel.
    """
    n = tmax.shape[0]
    spring = (doy >= 91) & (doy <= 151)
    greenup = 140.0 - 2.5 * (tmean[spring].mean(axis=0) - tmean[spring].mean())
    cum_p = np.vstack([np.zeros((1, tmax.shape[1])), np.cumsum(precip, axis=0)])
    expected = expected_daily_precip(doy, lo)
    cum_e = np.vstack([np.zeros((1, tmax.shape[1])), np.cumsum(expected, axis=0)])
    hot = np.vstack([np.zeros((1, tmax.shape[1])), np.cumsum(tmax > 32.0, axis=0)])
    comp_doy = 1 + NDVI_STEP_DAYS * np.arange(N_COMPOSITES)
    out = np.empty((N_COMPOSITES, tmax.shape[1]))
    damage = np.ones(tmax.shape[1])
    for i, d in enumerate(comp_doy):
        end = min(int(d) + NDVI_STEP_DAYS - 1, n)
        start = max(end - 60, 0)
        ratio = (cum_p[end] - cum_p[start]) / np.maximum(cum_e[end] - cum_e[start], 1e-6)
        f_water = 1.0 / (1.0 + np.exp(-6.0 * (ratio - 0.55)))
        f_heat = np.exp(-0.035 * (hot[end] - hot[max(end - 30, 0)]))
        if d >= 130:
            damage = np.minimum(damage, 0.25 + 0.75 * f_water * f_heat)
        pheno = 1.0 / (1.0 + np.exp(-(d - greenup) / 8.0)) * (1.0 - 1.0 / (1.0 + np.exp(-(d - 268.0) / 9.0)))
        out[i] = 0.17 + 0.68 * soil * pheno * damage
    out += rng.normal(0, 0.02, size=out.shape)
    out = np.clip(out, -0.1, 0.95)
    out[rng.random(out.shape) < 0.03] = np.nan  # cloud gaps
    return out


def _template(spec: CubeSpec) -> xr.Dataset:
    import dask.array as da

    lat, lon = grid(spec)
    time = time_index(spec.years)
    ntime = len(ndvi_time_index(spec.years))
    chunks = (spec.chunk_time, spec.chunk_lat, spec.chunk_lon)
    shape = (len(time), lat.size, lon.size)
    data = {v: (("time", "lat", "lon"), da.zeros(shape, chunks=chunks, dtype="float32")) for v in ("tmax", "tmin", "precip")}
    data["ndvi"] = (
        ("time_ndvi", "lat", "lon"),
        da.zeros((ntime, lat.size, lon.size), chunks=(ntime, spec.chunk_lat, spec.chunk_lon), dtype="float32"),
    )
    ds = xr.Dataset(data, coords={"time": time, "time_ndvi": ndvi_time_index(spec.years), "lat": lat, "lon": lon})
    ds["tmax"].attrs = {"units": "degC", "long_name": "daily maximum 2 m temperature (SYNTHETIC)"}
    ds["tmin"].attrs = {"units": "degC", "long_name": "daily minimum 2 m temperature (SYNTHETIC)"}
    ds["precip"].attrs = {"units": "mm/day", "long_name": "daily precipitation (SYNTHETIC)"}
    ds["ndvi"].attrs = {"units": "1", "long_name": "8-day composite NDVI (SYNTHETIC)"}
    ds.attrs = {"title": "SYNTHETIC agri-climate cube", "crs": "EPSG:4326", "synthetic": "true"}
    return ds


def write_cube(spec: CubeSpec, path: Path, seed: int) -> dict:
    """Write the full cube to Zarr one spatial tile at a time (all years per tile). Returns size info."""
    import shutil

    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    tmpl = _template(spec)
    encoding = {v: {"chunks": tmpl[v].data.chunksize} for v in tmpl.data_vars}
    tmpl.to_zarr(path, compute=False, encoding=encoding, mode="w", consolidated=True)
    lat, lon = grid(spec)
    for i0 in range(0, spec.n_lat, spec.chunk_lat):
        for j0 in range(0, spec.n_lon, spec.chunk_lon):
            la, lo = lat[i0 : i0 + spec.chunk_lat], lon[j0 : j0 + spec.chunk_lon]
            parts = [generate_year(y, la, lo, seed, (i0, j0)) for y in spec.years]
            tile = xr.Dataset(
                {v: (("time", "lat", "lon"), np.concatenate([p[v] for p in parts])) for v in ("tmax", "tmin", "precip")}
                | {"ndvi": (("time_ndvi", "lat", "lon"), np.concatenate([p["ndvi"] for p in parts]))}
            )
            region = {
                "time": slice(None),
                "time_ndvi": slice(None),
                "lat": slice(i0, i0 + la.size),
                "lon": slice(j0, j0 + lo.size),
            }
            tile.to_zarr(path, region=region, mode="r+")
            del parts, tile
    nbytes = int(sum(tmpl[v].nbytes for v in tmpl.data_vars))
    disk = int(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()))
    return {"path": str(path), "uncompressed_bytes": nbytes, "disk_bytes": disk, "dims": dict(tmpl.sizes)}
