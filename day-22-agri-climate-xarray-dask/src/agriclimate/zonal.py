"""Field polygons (SYNTHETIC) and raster-vector zonal statistics with rioxarray + rasterio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.features import rasterize
from shapely import affinity
from shapely.geometry import Polygon, box, mapping, shape

from .config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN


def make_fields(n: int, seed: int) -> list[dict]:
    """n non-overlapping field polygons: a rotated, jittered rectangle clipped to its own cell of a jittered grid."""
    rng = np.random.default_rng([seed, 5])
    ncol = int(np.ceil(np.sqrt(n * 1.5)))
    nrow = int(np.ceil(n / ncol))
    dy = (LAT_MAX - LAT_MIN) / nrow
    dx = (LON_MAX - LON_MIN) / ncol
    cells = rng.choice(nrow * ncol, size=n, replace=False)
    fields = []
    for fid, c in enumerate(sorted(cells), start=1):
        r, q = divmod(int(c), ncol)
        cell = box(LON_MIN + q * dx, LAT_MIN + r * dy, LON_MIN + (q + 1) * dx, LAT_MIN + (r + 1) * dy)
        cx = cell.centroid.x + rng.uniform(-0.2, 0.2) * dx
        cy = cell.centroid.y + rng.uniform(-0.2, 0.2) * dy
        w = rng.uniform(0.35, 0.7) * dx
        h = rng.uniform(0.35, 0.7) * dy
        corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2 + rng.normal(0, 0.04 * min(dx, dy), (4, 2))
        poly = affinity.rotate(Polygon(corners + [cx, cy]), rng.uniform(-30, 30), origin=(cx, cy))
        poly = poly.intersection(cell)
        fields.append({"field_id": fid, "geometry": poly})
    return fields


def fields_to_geojson(fields: list[dict], path: Path) -> None:
    feats = [
        {
            "type": "Feature",
            "properties": {"field_id": f["field_id"], "synthetic": True},
            "geometry": json.loads(json.dumps(mapping(f["geometry"]))),
        }
        for f in fields
    ]
    fc = {"type": "FeatureCollection", "name": "SYNTHETIC field polygons (EPSG:4326)", "features": feats}
    Path(path).write_text(json.dumps(fc, separators=(",", ":")))


def fields_from_geojson(path: Path) -> list[dict]:
    fc = json.loads(Path(path).read_text())
    return [{"field_id": f["properties"]["field_id"], "geometry": shape(f["geometry"])} for f in fc["features"]]


def with_spatial_ref(da: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    return da.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326")


def label_raster(fields: list[dict], template: xr.DataArray) -> tuple[xr.DataArray, int]:
    """Burn field ids into the template grid (cell-centre rule). Fields hitting no cell centre get their centroid cell.

    Returns the label raster (0 = no field) and how many fields needed the centroid fallback.
    """
    t = with_spatial_ref(template)
    transform = t.rio.transform()
    shp = (t.sizes["lat"], t.sizes["lon"])
    labels = rasterize(
        [(mapping(f["geometry"]), f["field_id"]) for f in fields],
        out_shape=shp,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="int32",
    )
    present = set(np.unique(labels)) - {0}
    lat = template["lat"].values
    lon = template["lon"].values
    fallback = 0
    for f in fields:
        if f["field_id"] not in present:
            c = f["geometry"].centroid
            i = int(np.abs(lat - c.y).argmin())
            j = int(np.abs(lon - c.x).argmin())
            if labels[i, j] == 0:
                labels[i, j] = f["field_id"]
                fallback += 1
    da = xr.DataArray(labels, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="field_id")
    return da, fallback


def zonal_mean(feats: xr.Dataset, labels: xr.DataArray) -> pd.DataFrame:
    """Mean of every feature over each field's cells, per year -> long table (field_id, year, features...)."""
    masked = feats.where(labels > 0)
    grouped = masked.groupby(labels.where(labels > 0)).mean()
    grouped = grouped.rename({"field_id": "fid"}) if "field_id" in grouped.dims else grouped
    df = grouped.to_dataframe().reset_index()
    key = "fid" if "fid" in df.columns else "field_id"
    df = df.rename(columns={key: "field_id"})
    df["field_id"] = df["field_id"].astype(int)
    return df.sort_values(["field_id", "year"]).reset_index(drop=True)


def field_table(fields: list[dict], labels: xr.DataArray) -> pd.DataFrame:
    counts = pd.Series(labels.values.ravel()).value_counts()
    rows = []
    for f in fields:
        c = f["geometry"].centroid
        rows.append(
            {
                "field_id": f["field_id"],
                "lat": c.y,
                "lon": c.x,
                "area_deg2": f["geometry"].area,
                "n_cells": int(counts.get(f["field_id"], 0)),
            }
        )
    return pd.DataFrame(rows)


def clip_mean(da: xr.DataArray, geom) -> float:
    """Cross-check: mean of a 2-D raster inside one polygon via rioxarray.clip (cell-centre rule)."""
    clipped = with_spatial_ref(da).rio.clip([mapping(geom)], crs="EPSG:4326", all_touched=False, drop=True)
    return float(clipped.mean(skipna=True))
