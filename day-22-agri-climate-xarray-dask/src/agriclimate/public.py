"""Optional public path: ERA5 daily climate from the Open-Meteo archive API (no key), sampled on a coarse lattice.

If the API answers, tmax / tmin / precipitation are fetched on a lattice over the study box and bilinearly
interpolated onto the cube grid; NDVI and yields stay SYNTHETIC (driven by that climate). If the API is not
reachable, the caller falls back to the seeded synthetic generator. Either way the path used is recorded.
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr

from .config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN

URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = {"temperature_2m_max": "tmax", "temperature_2m_min": "tmin", "precipitation_sum": "precip"}


def _get(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (fixed https URL)
        return r.read()


def probe(timeout: float = 10.0) -> tuple[bool, str]:
    url = f"{URL}?latitude=42&longitude=-93&start_date=2020-07-01&end_date=2020-07-02&daily=temperature_2m_max"
    try:
        payload = json.loads(_get(url, timeout))
        return ("daily" in payload), "ok"
    except Exception as exc:  # network errors, proxy 403, bad JSON
        return False, f"{type(exc).__name__}: {str(exc)[:120]}"


def lattice(n_lat: int = 5, n_lon: int = 7) -> tuple[np.ndarray, np.ndarray]:
    return np.linspace(LAT_MAX, LAT_MIN, n_lat), np.linspace(LON_MIN, LON_MAX, n_lon)


def parse_response(payload: list[dict] | dict, lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    """Open-Meteo multi-location JSON (row-major over the lattice) -> Dataset(time, lat, lon)."""
    items = payload if isinstance(payload, list) else [payload]
    if len(items) != lat.size * lon.size:
        raise ValueError(f"expected {lat.size * lon.size} locations, got {len(items)}")
    time = pd.to_datetime(items[0]["daily"]["time"])
    data = {}
    for key, name in VARS.items():
        arr = np.array([np.asarray(it["daily"][key], dtype="float64") for it in items])
        arr = arr.reshape(lat.size, lon.size, len(time)).transpose(2, 0, 1)
        data[name] = (("time", "lat", "lon"), arr)
    ds = xr.Dataset(data, coords={"time": time, "lat": lat, "lon": lon})
    ds["precip"] = ds["precip"].fillna(0.0).clip(min=0)
    return ds


def fetch_era5(years: tuple[int, ...], timeout: float = 60.0) -> xr.Dataset:
    lat, lon = lattice()
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    q = (
        f"?latitude={','.join(f'{v:.3f}' for v in la.ravel())}&longitude={','.join(f'{v:.3f}' for v in lo.ravel())}"
        f"&start_date={years[0]}-01-01&end_date={years[-1]}-12-31&daily={','.join(VARS)}&timezone=UTC"
    )
    return parse_response(json.loads(_get(URL + q, timeout)), lat, lon)


def interpolate_to_grid(coarse: xr.Dataset, lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    return coarse.interp(lat=lat, lon=lon, method="linear", kwargs={"fill_value": "extrapolate"})
