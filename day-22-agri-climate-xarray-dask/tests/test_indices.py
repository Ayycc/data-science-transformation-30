import numpy as np
import xarray as xr
from scipy import stats

from agriclimate.indices import (
    compute_climate_features,
    daily_gdd,
    fit_gamma,
    gamma_params,
    growing_season_gdd,
    heat_days,
    jja_precip,
    ndvi_season,
    perturb,
    seasonal_sum,
    seasonal_sum_resample,
    spi,
    spi_transform,
)


def test_daily_gdd_hand_values():
    tx = xr.DataArray([35.0, 20.0, 8.0, 25.0])
    tn = xr.DataArray([20.0, 5.0, 2.0, 15.0])
    # (min(35,30)+20)/2-10=15 ; (20+10)/2-10=5 ; (10+10)/2-10=0 ; (25+15)/2-10=10
    np.testing.assert_allclose(daily_gdd(tx, tn).values, [15, 5, 0, 10])


def test_seasonal_sum_matches_resample_and_numpy(cube):
    g = daily_gdd(cube["tmax"], cube["tmin"])
    a = seasonal_sum(g, (4, 9)).compute()
    b = seasonal_sum_resample(g, (4, 9)).compute()
    np.testing.assert_allclose(a.values, b.transpose(*a.dims).values, rtol=1e-5)
    gl = g.load()
    y0 = int(a["year"][0])
    m = (gl["time"].dt.year == y0) & (gl["time"].dt.month >= 4) & (gl["time"].dt.month <= 9)
    np.testing.assert_allclose(a.sel(year=y0).values, gl.where(m, 0).sum("time").values, rtol=1e-4)


def test_dask_and_in_memory_give_same_features(cube):
    lazy, _, tasks = compute_climate_features(cube, num_workers=2)
    eager, _, _ = compute_climate_features(cube.load(), num_workers=2)
    for v in lazy.data_vars:
        np.testing.assert_allclose(lazy[v].values, eager[v].values, rtol=1e-5)
    assert tasks["gdd"] > 0 and set(lazy.data_vars) == {"gdd", "heat_days", "precip_jja", "spi3_aug"}


def test_heat_days_and_jja_ranges(cube):
    h = heat_days(cube).compute()
    p = jja_precip(cube).compute()
    assert float(h.min()) >= 0 and float(h.max()) <= 92
    assert 50 < float(p.mean()) < 800


def test_fit_gamma_recovers_parameters():
    rng = np.random.default_rng(0)
    x = rng.gamma(4.0, 50.0, size=(3, 4000))
    a, b, q = fit_gamma(x)
    np.testing.assert_allclose(a, 4.0, rtol=0.08)
    np.testing.assert_allclose(b, 50.0, rtol=0.08)
    assert np.all(q == 0)


def test_spi_is_standard_normal_and_scenario_lowers_it():
    rng = np.random.default_rng(1)
    x = rng.gamma(3.0, 80.0, size=5000)
    a, b, q = fit_gamma(x)
    s = spi_transform(x, a, b, q)
    assert abs(s.mean()) < 0.05 and abs(s.std() - 1) < 0.05
    assert stats.kstest(s, "norm").pvalue > 0.01
    assert (spi_transform(0.8 * x, a, b, q) < s).all()


def test_spi_with_zeros():
    x = np.array([0.0, 0.0, 10, 20, 30, 40, 50, 60, 70, 80])
    a, b, q = fit_gamma(x)
    assert abs(q - 0.2) < 1e-12
    s = spi_transform(x, a, b, q)
    assert np.all(np.diff(s[1:]) > 0)


def test_spi_dask_baseline_params_for_scenario(cube):
    p = jja_precip(cube)
    params = gamma_params(p)
    base = spi(p, params).compute()
    scen = spi(jja_precip(perturb(cube, 2.0, 0.8)), params).compute()
    assert float(scen.mean()) < float(base.mean()) - 0.3


def test_perturb_shifts_gdd(cube):
    base = growing_season_gdd(cube).compute()
    warm = growing_season_gdd(perturb(cube, 2.0, 1.0)).compute()
    assert float((warm - base).mean()) > 150


def test_ndvi_season(cube):
    nd = ndvi_season(cube).compute()
    assert set(nd.data_vars) == {"ndvi_int", "ndvi_peak"}
    assert float(nd["ndvi_peak"].mean()) > 0.5
    assert np.isfinite(nd["ndvi_int"]).all()
