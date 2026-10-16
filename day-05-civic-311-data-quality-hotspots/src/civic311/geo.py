"""City geometry (simplified Chicago outline), neighbourhoods, wards and H3 helpers."""

from __future__ import annotations

import h3
import numpy as np
import shapely
from shapely.geometry import MultiPoint, Point, Polygon

from .config import N_NEIGHBOURHOODS, N_WARDS

# A simplified, hand-digitised approximation of the City of Chicago outline (lon, lat).
# It is NOT the official boundary; it only gives the synthetic city a realistic shape and lakefront.
CITY_LONLAT: list[tuple[float, float]] = [
    (-87.524, 41.645), (-87.620, 41.645), (-87.680, 41.690), (-87.740, 41.735), (-87.800, 41.750),
    (-87.800, 41.820), (-87.750, 41.860), (-87.810, 41.900), (-87.810, 41.950), (-87.940, 41.960),
    (-87.940, 42.000), (-87.810, 42.000), (-87.710, 42.020), (-87.670, 42.023), (-87.655, 41.998),
    (-87.638, 41.960), (-87.625, 41.920), (-87.612, 41.890), (-87.607, 41.860), (-87.580, 41.810),
    (-87.567, 41.780), (-87.525, 41.735),
]  # fmt: skip
CITY = Polygon(CITY_LONLAT)
shapely.prepare(CITY)
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = CITY.bounds

# Planted ground truth for the hotspot analysis (SYNTHETIC): resolution-time multipliers inside circles.
# (name, lon, lat, radius_km, multiplier)
PLANTED_ZONES: list[tuple[str, float, float, float, float]] = [
    ("SLOW-South", -87.660, 41.790, 1.8, 2.2),
    ("SLOW-West", -87.730, 41.880, 1.5, 2.2),
    ("SLOW-Northwest", -87.700, 41.965, 1.5, 2.2),
    ("FAST-Loop", -87.635, 41.880, 1.5, 0.5),
]

KM_PER_DEG_LAT = 111.32


def km_per_deg_lon(lat: float = 41.85) -> float:
    return KM_PER_DEG_LAT * float(np.cos(np.radians(lat)))


def in_city(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Vectorised point-in-polygon; NaN coordinates return False."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    ok = np.isfinite(lon) & np.isfinite(lat)
    out = np.zeros(lon.shape, dtype=bool)
    out[ok] = shapely.contains_xy(CITY, lon[ok], lat[ok])
    return out


def in_bbox(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    with np.errstate(invalid="ignore"):
        return (lon >= LON_MIN) & (lon <= LON_MAX) & (lat >= LAT_MIN) & (lat <= LAT_MAX)


def zone_multiplier(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True resolution-time multiplier from the planted zones and the zone index (-1 = none)."""
    mult = np.ones(len(lon))
    idx = np.full(len(lon), -1)
    for k, (_, zlon, zlat, r_km, m) in enumerate(PLANTED_ZONES):
        d = np.hypot((lon - zlon) * km_per_deg_lon(zlat), (lat - zlat) * KM_PER_DEG_LAT)
        inside = d <= r_km
        mult[inside] = m
        idx[inside] = k
    return mult, idx


def _seeds_in_city(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts: list[tuple[float, float]] = []
    while len(pts) < n:
        lon = rng.uniform(LON_MIN, LON_MAX, 4 * n)
        lat = rng.uniform(LAT_MIN, LAT_MAX, 4 * n)
        keep = in_city(lon, lat)
        pts.extend(zip(lon[keep], lat[keep], strict=True))
    return np.array(pts[:n])


WARD_SEEDS = _seeds_in_city(N_WARDS, 501)
NEIGHBOURHOOD_SEEDS = _seeds_in_city(N_NEIGHBOURHOODS, 777)


def nearest_seed(lon: np.ndarray, lat: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Index of the nearest seed in a locally isotropic (km) frame; this is the Voronoi assignment."""
    from scipy.spatial import cKDTree

    k = km_per_deg_lon()
    tree = cKDTree(np.c_[seeds[:, 0] * k, seeds[:, 1] * KM_PER_DEG_LAT])
    _, i = tree.query(np.c_[np.asarray(lon) * k, np.asarray(lat) * KM_PER_DEG_LAT])
    return i


def neighbourhood_polygons() -> list[Polygon]:
    """Voronoi polygons of the neighbourhood seeds (in the km frame), clipped to the city, in seed order."""
    k = km_per_deg_lon()
    pts = MultiPoint([Point(x * k, y * KM_PER_DEG_LAT) for x, y in NEIGHBOURHOOD_SEEDS])
    city_km = shapely.transform(CITY, lambda c: c * np.array([k, KM_PER_DEG_LAT]))
    cells = shapely.voronoi_polygons(pts, extend_to=city_km.buffer(5)).geoms
    out: list[Polygon | None] = [None] * len(NEIGHBOURHOOD_SEEDS)
    for cell in cells:
        for i, (x, y) in enumerate(NEIGHBOURHOOD_SEEDS):
            if out[i] is None and cell.contains(Point(x * k, y * KM_PER_DEG_LAT)):
                clipped = cell.intersection(city_km)
                out[i] = shapely.transform(clipped, lambda c: c / np.array([k, KM_PER_DEG_LAT]))
                break
    return out  # type: ignore[return-value]


def neighbourhood_income() -> np.ndarray:
    """SYNTHETIC median household income per neighbourhood: higher north and near the lake, plus noise."""
    rng = np.random.default_rng(778)
    lon, lat = NEIGHBOURHOOD_SEEDS[:, 0], NEIGHBOURHOOD_SEEDS[:, 1]
    north = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    lake = 1 - (lon - LON_MIN) / (LON_MAX - LON_MIN)
    z = 1.2 * north - 0.9 * lake + rng.normal(0, 0.35, len(lon))
    return np.round(np.exp(10.9 + 0.6 * z), -2)


def latlng_to_cells(lat: np.ndarray, lon: np.ndarray, res: int) -> np.ndarray:
    f = h3.latlng_to_cell
    return np.array([f(a, b, res) for a, b in zip(lat.tolist(), lon.tolist(), strict=True)], dtype=object)


def cell_boundary_lonlat(cell: str) -> list[tuple[float, float]]:
    return [(lng, lat) for lat, lng in h3.cell_to_boundary(cell)]
