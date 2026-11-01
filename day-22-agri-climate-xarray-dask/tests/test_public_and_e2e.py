import json

import numpy as np

from agriclimate import public
from agriclimate.pipeline import run


def test_parse_open_meteo_multi_location():
    lat, lon = public.lattice(2, 3)
    items = []
    for k in range(6):
        items.append({"daily": {"time": ["2020-07-01", "2020-07-02"], "temperature_2m_max": [30 + k, 31 + k],
                                "temperature_2m_min": [18, 19], "precipitation_sum": [0.0, None]}})  # fmt: skip
    ds = public.parse_response(json.loads(json.dumps(items)), lat, lon)
    assert ds["tmax"].shape == (2, 2, 3)
    assert float(ds["tmax"].isel(time=0, lat=1, lon=2)) == 35
    assert float(ds["precip"].isel(time=1).sum()) == 0.0
    fine = public.interpolate_to_grid(ds, np.linspace(44, 40, 5), np.linspace(-96, -90, 7))
    assert np.isfinite(fine["tmax"]).all()


def test_end_to_end_small(tmp_path, cfg):
    m = run(cfg, tmp_path, try_public=False, out_of_core=True)
    assert m["data"]["path_used"] == "synthetic_generator"
    cv = m["cv"]
    assert cv["climate_only__spatial"]["rmse"] > 0
    assert m["scenario"]["model"]["mean_change_pct"] < 0
    assert m["scenario"]["truth"]["mean_change_pct"] < 0
    ooc = m["performance"]["out_of_core"]["summary"]
    assert ooc["chunked_compute_status"] == "succeeded"
    for f in ("metrics.json", "report.md", "performance.md", "drought_exposure_map.html", "field_risk.csv"):
        assert (tmp_path / "results" / f).exists()
    assert len(list((tmp_path / "results" / "figures").glob("*.png"))) == 6
    assert not (tmp_path / "data" / "big_cube.zarr").exists()
