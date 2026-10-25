"""Station and H3-cell spatial features (GeoPandas + H3), elevation from public DEM tiles, OSM probe."""

from __future__ import annotations

import io
import math

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from . import config as C
from .data import fetch

UTM = "EPSG:32618"  # UTM zone 18N, metres, covers Washington DC


def haversine_matrix(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distances in metres, shape (len(lat1), len(lat2))."""
    p1, p2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dphi = p2 - p1
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6_371_000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def to_gdf(stations: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(stations.copy(), geometry=gpd.points_from_xy(stations["lon"], stations["lat"]), crs="EPSG:4326")


# --------------------------------------------------------------------------------------------- elevation
def _tile_xy(lat: float, lon: float, z: int) -> tuple[float, float]:
    n = 2**z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def elevation_from_tiles(lat: np.ndarray, lon: np.ndarray, z: int = C.TERRAIN_ZOOM) -> tuple[np.ndarray | None, str]:
    """Sample AWS Terrain Tiles (terrarium PNG: h = R*256 + G + B/256 - 32768). Tiles are cached in data/raw."""
    from PIL import Image

    out = np.full(len(lat), np.nan)
    tiles: dict[tuple[int, int], np.ndarray | None] = {}
    status = "not attempted"
    for i, (la, lo) in enumerate(zip(lat, lon, strict=True)):
        fx, fy = _tile_xy(float(la), float(lo), z)
        key = (int(fx), int(fy))
        if key not in tiles:
            content, status = fetch(
                C.TERRAIN_URL.format(z=z, x=key[0], y=key[1]), C.DATA_RAW / "terrain" / f"{z}_{key[0]}_{key[1]}.png", timeout=20
            )
            if content is None:
                return None, status
            rgb = np.asarray(Image.open(io.BytesIO(content)).convert("RGB"), dtype=float)
            tiles[key] = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
        px = min(255, int((fx - key[0]) * 256))
        py = min(255, int((fy - key[1]) * 256))
        out[i] = tiles[key][py, px]
    return out, f"{status}; {len(tiles)} tiles at zoom {z}"


# --------------------------------------------------------------------------------------------- OSM probe
def probe_osm_pois(stations: pd.DataFrame, timeout: int = 10) -> tuple[pd.Series | None, str]:
    """Count OSM amenities within 400 m of each station via OSMnx/Overpass. Returns (counts or None, status)."""
    _, status = fetch("https://overpass-api.de/api/status", timeout=timeout)
    if status.startswith("unreachable"):
        return None, f"Overpass API {status}"
    try:
        import osmnx as ox

        ox.settings.requests_timeout = timeout
        ox.settings.use_cache = False
        lat0, lon0 = float(stations["lat"].median()), float(stations["lon"].median())
        pois = ox.features_from_point((lat0, lon0), tags={"amenity": True, "shop": True}, dist=15000)
    except Exception as exc:  # noqa: BLE001
        return None, f"unreachable ({type(exc).__name__}: {str(exc)[:120]})"
    pts = pois.to_crs(UTM).geometry.centroid
    st = to_gdf(stations).to_crs(UTM)
    buf = gpd.GeoDataFrame(geometry=st.buffer(400), crs=UTM)
    joined = gpd.sjoin(gpd.GeoDataFrame(geometry=pts, crs=UTM), buf, predicate="within")
    counts = joined.groupby("index_right").size().reindex(range(len(stations)), fill_value=0)
    return pd.Series(counts.to_numpy(), index=stations.index), f"reachable ({len(pois)} features)"


# --------------------------------------------------------------------------------------------- features
def add_spatial_features(stations: pd.DataFrame, res: int = C.H3_RES, try_osm: bool = True) -> tuple[pd.DataFrame, dict]:
    """Attach H3 cell, transit proximity, elevation, density and POI features to each station."""
    st = stations.copy().reset_index(drop=True)
    prov: dict = {}
    st["h3"] = [h3.latlng_to_cell(la, lo, res) for la, lo in zip(st["lat"], st["lon"], strict=True)]

    # transit proximity: distance to the nearest Metrorail-adjacent dock (station names containing "Metro")
    g = to_gdf(st).to_crs(UTM)
    metro = g[st["is_metro"].astype(bool)][["geometry"]]
    if len(metro):
        near = gpd.sjoin_nearest(g[["geometry"]], metro, distance_col="metro_dist_m")
        st["metro_dist_m"] = near.groupby(level=0)["metro_dist_m"].min().reindex(st.index).to_numpy()
    else:
        st["metro_dist_m"] = np.nan
    prov["transit"] = f"{int(st['is_metro'].sum())} Metrorail-adjacent docks identified from station names"

    # elevation proxy
    elev, status = elevation_from_tiles(st["lat"].to_numpy(), st["lon"].to_numpy())
    if elev is not None:
        st["elevation_m"] = np.round(elev, 1)
        prov["elevation"] = f"AWS Terrain Tiles (public DEM): {status}"
    elif "elevation_m" in st:
        prov["elevation"] = f"committed sample (tiles {status})"
    else:
        st["elevation_m"] = 0.0
        prov["elevation"] = f"unavailable, set to 0 ({status})"

    # H3 neighbourhoods
    cell_of = st["h3"].to_numpy()
    by_cell: dict[str, list[int]] = {}
    for i, c in enumerate(cell_of):
        by_cell.setdefault(c, []).append(i)

    def ring_members(cell: str, k: int) -> list[int]:
        return [j for c in h3.grid_disk(cell, k) for j in by_cell.get(c, [])]

    elev_arr = st["elevation_m"].to_numpy(float)
    st["stations_in_cell"] = [len(by_cell[c]) for c in cell_of]
    st["stations_ring1"] = [len(ring_members(c, 1)) for c in cell_of]
    st["stations_ring2"] = [len(ring_members(c, 2)) for c in cell_of]
    st["elev_rel_ring2"] = [elev_arr[i] - elev_arr[ring_members(c, 2)].mean() for i, c in enumerate(cell_of)]
    xy = np.c_[g.geometry.x, g.geometry.y]
    core = np.average(xy, axis=0, weights=st["activity"].to_numpy(float) if "activity" in st else None)
    st["dist_core_m"] = np.hypot(*(xy - core).T)
    st["x_m"], st["y_m"] = xy[:, 0], xy[:, 1]

    pois, status = probe_osm_pois(st) if try_osm else (None, "skipped")
    prov["osm"] = status
    if pois is not None:
        st["poi_400m"] = pois.to_numpy()
        prov["poi"] = "OSM amenity+shop count within 400 m (OSMnx)"
    else:
        st["poi_400m"] = st["stations_ring2"]
        prov["poi"] = "OSM unreachable: POI density proxied by dock density (stations in the H3 k=2 ring)"
    return st, prov


def neighbour_lists(stations: pd.DataFrame, k: int = 1, min_neighbours: int = 3) -> list[np.ndarray]:
    """Spatial weights: other stations in the H3 k-ring; falls back to the nearest stations if too few."""
    cell_of = stations["h3"].to_numpy()
    by_cell: dict[str, list[int]] = {}
    for i, c in enumerate(cell_of):
        by_cell.setdefault(c, []).append(i)
    d = haversine_matrix(stations["lat"].to_numpy(), stations["lon"].to_numpy(), stations["lat"].to_numpy(), stations["lon"].to_numpy())
    out = []
    for i, c in enumerate(cell_of):
        nb = [j for cc in h3.grid_disk(c, k) for j in by_cell.get(cc, []) if j != i]
        if len(nb) < min_neighbours:
            nb = list(np.argsort(d[i])[1 : min_neighbours + 1])
        out.append(np.asarray(nb, dtype=int))
    return out


def h3_cells_gdf(stations: pd.DataFrame, values: pd.DataFrame | None = None) -> gpd.GeoDataFrame:
    """Hexagon polygons for every occupied H3 cell, with station-level values aggregated per cell."""
    agg = stations.groupby("h3").agg(
        n_stations=("station_id", "size"), elevation_m=("elevation_m", "mean"), metro_dist_m=("metro_dist_m", "min")
    )
    if values is not None:
        agg = agg.join(values.groupby(stations["h3"]).sum(), how="left")
    polys = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in agg.index]
    return gpd.GeoDataFrame(agg.reset_index(), geometry=polys, crs="EPSG:4326")
