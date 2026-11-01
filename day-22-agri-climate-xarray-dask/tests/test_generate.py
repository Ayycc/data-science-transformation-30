from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from agriclimate.config import SEED, CubeSpec, RunConfig
from agriclimate.generate import drought_footprint, generate_year, grid

ROOT = Path(__file__).resolve().parents[1]


def test_generate_year_is_deterministic_and_physical():
    spec = CubeSpec(8, 12, (2012,))
    lat, lon = grid(spec)
    a = generate_year(2012, lat, lon, SEED, (0, 0))
    b = generate_year(2012, lat, lon, SEED, (0, 0))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert a["tmax"].shape == (366, 8, 12)  # leap year
    assert a["ndvi"].shape == (46, 8, 12)
    assert (a["tmax"] > a["tmin"]).all()
    assert (a["precip"] >= 0).all()
    nd = a["ndvi"][np.isfinite(a["ndvi"])]
    assert nd.min() >= -0.1 and nd.max() <= 0.95
    assert 0.0 < np.isnan(a["ndvi"]).mean() < 0.08  # cloud gaps


def test_grid_is_north_up_and_regular():
    lat, lon = grid(CubeSpec(80, 120, (2001,)))
    assert lat[0] > lat[-1] and lon[0] < lon[-1]
    np.testing.assert_allclose(np.diff(lat), -0.05, atol=1e-6)
    np.testing.assert_allclose(np.diff(lon), 0.05, atol=1e-6)


def test_drought_reduces_summer_precip_and_ndvi():
    lat, lon = grid(CubeSpec(20, 30, (2012,)))
    fp = drought_footprint(2012, lat, lon, SEED)
    core = fp > 0.6 * fp.max()
    doy = pd.date_range("2012-01-01", "2012-12-31").day_of_year.to_numpy()
    jja = (doy >= 152) & (doy <= 243)

    def summer_p(year):
        d = generate_year(year, lat, lon, SEED, (0, 0))
        return d["precip"][jja[: d["precip"].shape[0]]].sum(0)[core].mean(), np.nanmax(d["ndvi"], 0)[core].mean()

    p12, n12 = summer_p(2012)
    normal = [summer_p(y) for y in (2010, 2011, 2013, 2014)]
    assert p12 < 0.75 * np.mean([p for p, _ in normal])
    assert n12 < np.mean([n for _, n in normal])


def test_written_cube_matches_generator(cube, cfg):
    lat, lon = grid(cfg.cube)
    y = cfg.cube.years[0]
    d = generate_year(y, lat[: cfg.cube.chunk_lat], lon[: cfg.cube.chunk_lon], cfg.seed, (0, 0))
    stored = cube["tmax"].sel(time=str(y)).isel(lat=slice(0, cfg.cube.chunk_lat), lon=slice(0, cfg.cube.chunk_lon)).values
    np.testing.assert_array_equal(stored, d["tmax"])
    assert cube["tmax"].data.chunksize == (cfg.cube.chunk_time, cfg.cube.chunk_lat, cfg.cube.chunk_lon)


def test_committed_sample_reproduces_from_generator():
    """data/sample/cube_sample_2012_SYNTHETIC.nc is the first 8 x 8 cells of tile (0, 0) of the full cube in 2012."""
    path = ROOT / "data" / "sample" / "cube_sample_2012_SYNTHETIC.nc"
    sample = xr.open_dataset(path, engine="scipy")
    spec = RunConfig().cube
    lat, lon = grid(spec)
    d = generate_year(2012, lat[: spec.chunk_lat], lon[: spec.chunk_lon], SEED, (0, 0))
    for v in ("tmax", "tmin", "precip"):
        np.testing.assert_array_equal(sample[v].values, d[v][:, :8, :8])
    np.testing.assert_array_equal(np.isnan(sample["ndvi"].values), np.isnan(d["ndvi"][:, :8, :8]))
