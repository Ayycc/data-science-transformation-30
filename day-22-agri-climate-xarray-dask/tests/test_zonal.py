import numpy as np
import xarray as xr
from shapely.geometry import Point

from agriclimate.zonal import clip_mean, field_table, fields_from_geojson, fields_to_geojson, label_raster, make_fields, zonal_mean


def _template(cube):
    return cube["tmax"].isel(time=0)


def test_fields_do_not_overlap_and_roundtrip(tmp_path):
    fields = make_fields(30, 7)
    for i, a in enumerate(fields):
        for b in fields[i + 1 :]:
            assert a["geometry"].intersection(b["geometry"]).area < 1e-12
    fields_to_geojson(fields, tmp_path / "f.geojson")
    back = fields_from_geojson(tmp_path / "f.geojson")
    assert [f["field_id"] for f in back] == [f["field_id"] for f in fields]
    assert abs(back[3]["geometry"].area - fields[3]["geometry"].area) < 1e-12


def test_label_raster_cell_centres_inside_polygons(cube):
    fields = make_fields(25, 3)
    labels, _ = label_raster(fields, _template(cube))
    geoms = {f["field_id"]: f["geometry"] for f in fields}
    lat, lon = labels["lat"].values, labels["lon"].values
    hits = np.argwhere(labels.values > 0)
    inside = [geoms[int(labels.values[i, j])].buffer(1e-9).contains(Point(lon[j], lat[i])) for i, j in hits]
    assert np.mean(inside) > 0.9  # the rest are centroid fallbacks for fields smaller than a cell
    assert set(np.unique(labels.values)) - {0} == set(geoms)


def test_zonal_mean_matches_manual_and_rioxarray_clip(cube):
    fields = make_fields(25, 3)
    tmpl = _template(cube)
    labels, fallback = label_raster(fields, tmpl)
    rng = np.random.default_rng(0)
    vals = xr.DataArray(
        rng.normal(size=(2,) + labels.shape), coords={"year": [1, 2], "lat": labels.lat, "lon": labels.lon}, dims=("year", "lat", "lon")
    )
    df = zonal_mean(xr.Dataset({"x": vals}), labels)
    assert len(df) == 2 * len(fields)
    for fid in (1, 7, 20):
        m = labels.values == fid
        manual = vals.sel(year=2).values[m].mean()
        got = df.query("field_id == @fid and year == 2")["x"].item()
        np.testing.assert_allclose(got, manual)
    # rioxarray.clip on a field that hit at least one cell centre (no fallback) gives the same mean
    tab = field_table(fields, labels)
    fid = int(tab.sort_values("n_cells").iloc[-1]["field_id"])
    geom = next(f["geometry"] for f in fields if f["field_id"] == fid)
    np.testing.assert_allclose(clip_mean(vals.sel(year=1), geom), df.query("field_id == @fid and year == 1")["x"].item(), rtol=1e-9)
