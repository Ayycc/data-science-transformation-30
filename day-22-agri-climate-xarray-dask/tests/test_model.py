import numpy as np
import pandas as pd

from agriclimate.model import (
    add_true_yield,
    cross_validate,
    risk_metrics,
    spatial_blocks,
    spatial_year_residual,
    yield_response,
)


def _table(n_fields=60, years=range(2001, 2011), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    lat = rng.uniform(40, 44, n_fields)
    lon = rng.uniform(-96, -90, n_fields)
    for y in years:
        for i in range(n_fields):
            rows.append(
                {
                    "field_id": i + 1,
                    "year": y,
                    "lat": lat[i],
                    "lon": lon[i],
                    "gdd": rng.normal(1600, 120),
                    "heat_days": rng.poisson(12),
                    "precip_jja": rng.normal(330, 60),
                    "spi3_aug": rng.normal(),
                    "ndvi_int": rng.normal(80, 5),
                    "ndvi_peak": rng.normal(0.8, 0.03),
                    "soil": 1.0 + 0.1 * np.sin(lat[i]) * np.cos(lon[i]),
                }  # fmt: skip
            )
    return add_true_yield(pd.DataFrame(rows), 5)


def test_yield_response_directions():
    base = yield_response(1700, 10, 0.0, 1.0, 2010, 0.0)
    assert yield_response(1700, 30, 0.0, 1.0, 2010, 0.0) < base
    assert yield_response(1700, 10, -2.0, 1.0, 2010, 0.0) < base
    assert yield_response(1700, 10, 0.0, 1.1, 2010, 0.0) > base
    assert 8 < base < 13


def test_residual_is_spatially_smooth():
    lat = np.array([42.0, 42.01, 43.5])
    lon = np.array([-93.0, -93.01, -91.0])
    r = spatial_year_residual(lat, lon, np.array([2005, 2005, 2005]), 1)
    assert abs(r[0] - r[1]) < 0.05


def test_spatial_blocks():
    b = spatial_blocks(np.array([40.1, 40.9, 41.2]), np.array([-95.9, -95.1, -95.5]), 1.0)
    assert b[0] == b[1] != b[2]


def test_cross_validate_schemes_run():
    df = _table()
    feats = ["gdd", "heat_days", "precip_jja", "spi3_aug", "lat", "lon", "year"]
    r = cross_validate(df, feats, "random", 3, 2.0, 0)
    s = cross_validate(df, feats, "spatial", 3, 2.0, 0)
    assert r["rmse"] > 0 and s["rmse"] > 0 and r["scheme"] == "random" and s["scheme"] == "spatial"


def test_risk_metrics_known_values():
    tab = pd.DataFrame({"field_id": [1] * 100, "a": np.arange(100.0), "b": np.arange(100.0) * 0.5})
    r = risk_metrics(tab, "a", "b")
    assert r["baseline"]["mean_t_ha"] == 49.5
    np.testing.assert_allclose(r["baseline"]["p5_t_ha"], np.percentile(np.arange(100.0), 5))
    np.testing.assert_allclose(r["mean_change_pct"], -50.0)
    assert r["scenario"]["p_below_trigger"] > r["baseline"]["p_below_trigger"]
