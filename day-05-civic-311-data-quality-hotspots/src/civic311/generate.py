"""SYNTHETIC 311 request generator: a Chicago-shaped city with realistic, labelled data defects.

The raw extract is written exactly as a CRM export would look (every field a string). A separate truth
file (same row order) records the injected defect class, the true category and the true resolution-time
multiplier for every row; the pipeline never reads it, only the evaluation does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from . import geo
from .categories import CATEGORIES, JUNK_VALUES, variant_catalog
from .config import EXTRACT_TS, WINDOW_START, Scale, data_dir

WEEKDAY = np.array([1.15, 1.10, 1.05, 1.05, 1.00, 0.75, 0.65])
STORM_DAYS = ("2025-07-15", "2025-07-16")
STORM_BOOST = {"Water On Street": (8.0, 4.0), "Sewer Cave-In Inspection": (6.0, 3.0), "Tree Trim": (5.0, 2.5)}
ORIGINS = np.array(["Phone Call", "Internet", "Mobile App", "E-Mail", "Internal"])
ORIGIN_P = np.array([0.35, 0.25, 0.30, 0.05, 0.05])
CANCEL_P = 0.03
VARIANT_P = 0.28  # share of rows whose category arrives in a non-canonical spelling
LOW_INCOME_MULT = 1.15  # SYNTHETIC equity effect: bottom-income-quartile neighbourhoods wait 15% longer

# Row-level defect classes and their rates (share of all base rows); disjoint row sets.
DEFECT_RATES: dict[str, float] = {
    "COORDS_MISSING": 0.030,
    "COORDS_NULL_ISLAND": 0.004,
    "COORDS_SWAPPED": 0.003,
    "COORDS_OUTSIDE_CITY": 0.004,
    "CLOSED_BEFORE_CREATED": 0.008,
    "CREATED_IN_FUTURE": 0.002,
    "CLOSED_PLACEHOLDER": 0.003,
    "TIMESTAMP_UNPARSEABLE": 0.002,
    "COMPLETED_WITHOUT_CLOSE": 0.006,
    "OPEN_WITH_CLOSE": 0.002,
    "STATUS_INVALID": 0.002,
    "WARD_INVALID": 0.004,
    "SR_NUMBER_MALFORMED": 0.0015,
    "CATEGORY_UNMAPPABLE": 0.004,
    "DUPLICATE_RECORD": 0.005,
}

# Data-quality incidents (ground truth for KPI anomaly detection).
INCIDENTS = [
    {"id": "INC1", "start": "2025-04-08", "end": "2025-04-11", "defect": "COORDS_MISSING",
     "what": "mobile-app release drops coordinates on 55% of Mobile App requests", "share": 0.55},
    {"id": "INC2", "start": "2025-09-16", "end": "2025-09-17", "defect": "CLOSED_BEFORE_CREATED",
     "what": "ETL bug writes closed timestamps before created on 30% of closed requests", "share": 0.30},
    {"id": "INC3", "start": "2025-11-20", "end": "2025-11-22", "defect": "WARD_INVALID",
     "what": "CRM sync outage blanks the ward on 45% of requests", "share": 0.45},
]  # fmt: skip

TS_FMT = "%Y-%m-%d %H:%M:%S"
_NEAR_EDGE = geo.CITY.boundary.buffer(1e-4)


def _sample_points(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """60% from 12 dense clusters (sd ~1.5 km), 40% uniform; all inside the city polygon."""
    centres = geo.NEIGHBOURHOOD_SEEDS[rng.choice(len(geo.NEIGHBOURHOOD_SEEDS), 12, replace=False)]
    lon_out, lat_out = np.empty(0), np.empty(0)
    while len(lon_out) < n:
        m = int((n - len(lon_out)) * 1.5) + 100
        clustered = rng.random(m) < 0.6
        c = centres[rng.integers(0, len(centres), m)]
        lon = np.where(clustered, c[:, 0] + rng.normal(0, 1.5, m) / geo.km_per_deg_lon(),
                       rng.uniform(geo.LON_MIN, geo.LON_MAX, m))  # fmt: skip
        lat = np.where(clustered, c[:, 1] + rng.normal(0, 1.5, m) / geo.KM_PER_DEG_LAT,
                       rng.uniform(geo.LAT_MIN, geo.LAT_MAX, m))  # fmt: skip
        lon, lat = _as_exported(lon), _as_exported(lat)
        keep = geo.in_city(lon, lat)
        lon_out = np.r_[lon_out, lon[keep]]
        lat_out = np.r_[lat_out, lat[keep]]
    return lon_out[:n], lat_out[:n]


def _outside_city_points(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """60% inside the bounding box but outside the city (lake, suburbs); 40% far away (geocoder errors)."""
    lon, lat = np.empty(0), np.empty(0)
    while len(lon) < n:
        a = rng.uniform(geo.LON_MIN, geo.LON_MAX, 4 * n + 50)
        b = rng.uniform(geo.LAT_MIN, geo.LAT_MAX, 4 * n + 50)
        a, b = _as_exported(a), _as_exported(b)
        k = ~geo.in_city(a, b) & ~shapely.contains(_NEAR_EDGE, shapely.points(a, b))  # clearly outside
        lon, lat = np.r_[lon, a[k]], np.r_[lat, b[k]]
    lon, lat = lon[:n], lat[:n]
    far = rng.random(n) < 0.4
    places = np.array([(-74.006, 40.713), (-89.650, 39.781), (-87.633, 40.0), (-122.42, 37.77)])
    p = places[rng.integers(0, len(places), n)]
    lon[far] = p[far, 0] + rng.normal(0, 0.05, far.sum())
    lat[far] = p[far, 1] + rng.normal(0, 0.05, far.sum())
    return lon, lat


def _as_exported(x: np.ndarray) -> np.ndarray:
    """Round-trip through the 6-decimal text export so geometry tests see exactly what the pipeline reads."""
    return np.char.mod("%.6f", x).astype(float)


def _fmt_coord(x: np.ndarray) -> np.ndarray:
    return np.char.mod("%.6f", x).astype(object)


def generate(scale: Scale) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (raw extract as strings, truth table, manifest). Deterministic for a given scale."""
    rng = np.random.default_rng(scale.seed)
    days = pd.date_range(WINDOW_START, periods=365, freq="D")
    doy = days.dayofyear.to_numpy()
    cats = list(CATEGORIES)
    rate = np.zeros((len(days), len(cats)))
    for j, c in enumerate(cats):
        rate[:, j] = c.share * (1 + c.season_amp * np.cos(2 * np.pi * (doy - c.season_peak_doy) / 365))
    rate *= WEEKDAY[days.dayofweek.to_numpy()][:, None]
    for j, c in enumerate(cats):
        if c.name in STORM_BOOST:
            for d, boost in zip(STORM_DAYS, STORM_BOOST[c.name], strict=True):
                rate[days.get_loc(pd.Timestamp(d)), j] *= boost
    rate *= scale.n_target / rate.sum()
    counts = rng.poisson(rate)
    day_idx, cat_idx = np.nonzero(counts)
    reps = counts[day_idx, cat_idx]
    day_idx, cat_idx = np.repeat(day_idx, reps), np.repeat(cat_idx, reps)
    n = len(day_idx)

    # timestamps (seconds since window start)
    hour = np.where(rng.random(n) < 0.7, np.clip(rng.normal(13, 3.5, n), 0, 23.999), rng.uniform(0, 24, n))
    created_s = day_idx * 86400 + np.floor(hour * 3600)
    order = np.argsort(created_s, kind="stable")
    day_idx, cat_idx, created_s = day_idx[order], cat_idx[order], created_s[order]

    lon, lat = _sample_points(n, rng)
    zone_mult, zone_idx = geo.zone_multiplier(lon, lat)
    hood = geo.nearest_seed(lon, lat, geo.NEIGHBOURHOOD_SEEDS)
    income = geo.neighbourhood_income()
    low_income = income <= np.quantile(income, 0.25)
    inc_mult = np.where(low_income[hood], LOW_INCOME_MULT, 1.0)
    backlog = 1 + 0.15 * np.cos(2 * np.pi * (doy[day_idx] - 30) / 365)
    med = np.array([c.median_hours for c in cats])[cat_idx]
    sig = np.array([c.sigma for c in cats])[cat_idx]
    true_mult = zone_mult * inc_mult
    res_h = np.maximum(med * np.exp(sig * rng.standard_normal(n)) * true_mult * backlog, 0.25)
    canceled = rng.random(n) < CANCEL_P
    res_h[canceled] = rng.uniform(0.5, 72, canceled.sum())
    extract_s = (pd.Timestamp(EXTRACT_TS) - pd.Timestamp(WINDOW_START)).total_seconds()
    closed_s = created_s + np.round(res_h * 3600)
    is_open = closed_s > extract_s
    status = np.where(is_open, "Open", np.where(canceled, "Canceled", "Completed")).astype(object)
    closed_s = np.where(is_open, np.nan, closed_s)
    origin = ORIGINS[rng.choice(len(ORIGINS), n, p=ORIGIN_P)].astype(object)
    ward = (geo.nearest_seed(lon, lat, geo.WARD_SEEDS) + 1).astype(str).astype(object)
    dept = np.array([c.department for c in cats], dtype=object)[cat_idx]
    true_cat = np.array([c.name for c in cats], dtype=object)[cat_idx]

    # raw category spellings
    catalog = variant_catalog()
    sr_type = true_cat.copy()
    use_var = rng.random(n) < VARIANT_P
    for name, grp in catalog.dropna(subset=["true_category"]).groupby("true_category"):
        rows = np.nonzero(use_var & (true_cat == name))[0]
        sr_type[rows] = grp.raw_value.to_numpy()[rng.integers(0, len(grp), len(rows))]
    variant_kind = np.full(n, "canonical", dtype=object)
    kind_map = dict(zip(catalog.raw_value, catalog.kind, strict=True))
    variant_kind[use_var] = [kind_map[v] for v in sr_type[use_var]]

    # ---------------------------------------------------------------- inject row-level defects
    defect = np.full(n, "", dtype=object)
    incident = np.full(n, "", dtype=object)
    lat_s, lon_s = _fmt_coord(lat), _fmt_coord(lon)
    t0 = pd.Timestamp(WINDOW_START)
    created_str = (t0 + pd.to_timedelta(created_s, unit="s")).strftime(TS_FMT).to_numpy(dtype=object)
    closed_td = pd.to_timedelta(np.nan_to_num(closed_s), unit="s")
    closed_str = np.where(np.isnan(closed_s), "", (t0 + closed_td).strftime(TS_FMT).to_numpy(dtype=object))
    closed_str = closed_str.astype(object)

    def pick(name: str, rate_: float, cond: np.ndarray | None = None) -> np.ndarray:
        pool = defect == ""
        if cond is not None:
            pool &= cond
        cand = np.nonzero(pool)[0]
        k = min(int(round(rate_ * n)), len(cand))
        idx = np.sort(rng.choice(cand, k, replace=False))
        defect[idx] = name
        return idx

    def ts_str(sec: np.ndarray) -> np.ndarray:
        return (t0 + pd.to_timedelta(sec, unit="s")).strftime(TS_FMT).to_numpy(dtype=object)

    # incidents first (they define specific days)
    dates = days[day_idx]
    for inc in INCIDENTS:
        on = (dates >= inc["start"]) & (dates <= inc["end"])
        if inc["defect"] == "COORDS_MISSING":
            cond = on & (origin == "Mobile App")
        elif inc["defect"] == "CLOSED_BEFORE_CREATED":
            cond = on & (status == "Completed")
        else:
            cond = on
        idx = np.nonzero(cond & (rng.random(n) < inc["share"]))[0]
        defect[idx] = inc["defect"]
        incident[idx] = inc["id"]
        if inc["defect"] == "COORDS_MISSING":
            lat_s[idx] = ""
            lon_s[idx] = ""
        elif inc["defect"] == "CLOSED_BEFORE_CREATED":
            closed_str[idx] = ts_str(created_s[idx] - rng.uniform(3600, 6 * 3600, len(idx)))
        else:
            ward[idx] = ""

    completed = status == "Completed"
    i = pick("COORDS_MISSING", DEFECT_RATES["COORDS_MISSING"])
    lat_only = rng.random(len(i)) < 0.2
    lat_s[i] = ""
    lon_s[i[~lat_only]] = ""
    i = pick("COORDS_NULL_ISLAND", DEFECT_RATES["COORDS_NULL_ISLAND"])
    lat_s[i], lon_s[i] = "0.0", "0.0"
    i = pick("COORDS_SWAPPED", DEFECT_RATES["COORDS_SWAPPED"])
    lat_s[i], lon_s[i] = lon_s[i].copy(), lat_s[i].copy()
    i = pick("COORDS_OUTSIDE_CITY", DEFECT_RATES["COORDS_OUTSIDE_CITY"])
    a, b = _outside_city_points(len(i), rng)
    lon_s[i], lat_s[i] = _fmt_coord(a), _fmt_coord(b)
    i = pick("CLOSED_BEFORE_CREATED", DEFECT_RATES["CLOSED_BEFORE_CREATED"], completed)
    closed_str[i] = ts_str(created_s[i] - rng.uniform(3600, 30 * 86400, len(i)))
    i = pick("CREATED_IN_FUTURE", DEFECT_RATES["CREATED_IN_FUTURE"], completed)
    created_str[i] = ts_str(extract_s + rng.uniform(86400, 400 * 86400, len(i)))
    status[i], closed_str[i] = "Open", ""
    i = pick("CLOSED_PLACEHOLDER", DEFECT_RATES["CLOSED_PLACEHOLDER"], completed)
    closed_str[i] = np.where(rng.random(len(i)) < 0.5, "1900-01-01 00:00:00", "1970-01-01 00:00:00")
    i = pick("TIMESTAMP_UNPARSEABLE", DEFECT_RATES["TIMESTAMP_UNPARSEABLE"])
    created_str[i] = rng.choice(np.array(["N/A", "00/00/0000 00:00", "2025-13-45 25:61:00", "TBD"]), len(i))
    i = pick("COMPLETED_WITHOUT_CLOSE", DEFECT_RATES["COMPLETED_WITHOUT_CLOSE"], completed)
    closed_str[i] = ""
    i = pick("OPEN_WITH_CLOSE", DEFECT_RATES["OPEN_WITH_CLOSE"], completed)
    status[i] = "Open"
    i = pick("STATUS_INVALID", DEFECT_RATES["STATUS_INVALID"])
    status[i] = rng.choice(np.array(["DONE", "Closed?", "", "completed - dup", "Cancelled"]), len(i))
    i = pick("WARD_INVALID", DEFECT_RATES["WARD_INVALID"])
    ward[i] = rng.choice(np.array(["0", "99", "-1", "", "Ward 7", "51"]), len(i))
    sr_number = np.char.add("SR25-", np.char.zfill(np.arange(1, n + 1).astype(str), 8)).astype(object)
    i = pick("SR_NUMBER_MALFORMED", DEFECT_RATES["SR_NUMBER_MALFORMED"])
    forms = rng.integers(0, 4, len(i))
    sr_number[i] = [
        (f"SR25-{r % 1000}", f"sr25 {r + 1:07d}", "", f"25-{r + 1:08d}")[f] for r, f in zip(i, forms, strict=True)
    ]
    i = pick("CATEGORY_UNMAPPABLE", DEFECT_RATES["CATEGORY_UNMAPPABLE"])
    sr_type[i] = rng.choice(np.array([*JUNK_VALUES, ""], dtype=object), len(i))
    variant_kind[i] = "junk"

    raw = pd.DataFrame(
        {
            "sr_number": sr_number, "sr_type": sr_type, "owner_department": dept, "status": status,
            "created_date": created_str, "closed_date": closed_str, "origin": origin, "ward": ward,
            "latitude": lat_s, "longitude": lon_s,
        }
    )  # fmt: skip
    truth = pd.DataFrame(
        {
            "defect": defect, "incident": incident, "true_category": true_cat, "variant_kind": variant_kind,
            "true_multiplier": np.round(true_mult, 4), "zone": zone_idx, "neighbourhood": hood,
            "true_lon": np.round(lon, 6), "true_lat": np.round(lat, 6),
        }
    )  # fmt: skip

    # duplicates: exact copies of clean rows, placed right after the original
    clean_idx = np.nonzero(defect == "")[0]
    k = int(round(DEFECT_RATES["DUPLICATE_RECORD"] * n))
    dup_src = np.sort(rng.choice(clean_idx, k, replace=False))
    pos = np.r_[np.arange(n), dup_src + 0.5]
    order = np.argsort(pos, kind="stable")
    dup_truth = truth.iloc[dup_src].assign(defect="DUPLICATE_RECORD")
    raw = pd.concat([raw, raw.iloc[dup_src]], ignore_index=True).iloc[order].reset_index(drop=True)
    truth = pd.concat([truth, dup_truth], ignore_index=True).iloc[order].reset_index(drop=True)

    manifest = {
        "path_used": "synthetic",
        "scale": scale.name,
        "seed": scale.seed,
        "rows": len(raw),
        "base_requests": n,
        "duplicates_appended": k,
        "window": [WINDOW_START, EXTRACT_TS],
        "defect_rates": DEFECT_RATES,
        "incidents": INCIDENTS,
        "storm_days": list(STORM_DAYS),
        "planted_zones": [
            {"name": z[0], "lon": z[1], "lat": z[2], "radius_km": z[3], "multiplier": z[4]} for z in geo.PLANTED_ZONES
        ],
        "low_income_multiplier": LOW_INCOME_MULT,
        "defect_counts": {str(a): int(b) for a, b in truth.defect.replace("", "none").value_counts().items()},
    }
    return raw, truth, manifest


def file_paths(scale: Scale) -> dict[str, Path]:
    d = data_dir(scale)
    ext = ".csv.gz" if scale.name == "sample" else ".csv"
    return {"raw": d / f"requests_raw{ext}", "truth": d / f"truth{ext}", "manifest": d / "manifest.json"}


def write(scale: Scale, public_probe: dict | None = None) -> dict:
    raw, truth, manifest = generate(scale)
    p = file_paths(scale)
    p["raw"].parent.mkdir(parents=True, exist_ok=True)
    opts = {"compression": {"method": "gzip", "mtime": 0}} if scale.name == "sample" else {}
    raw.to_csv(p["raw"], index=False, **opts)
    truth.to_csv(p["truth"], index=False, **opts)
    manifest["files"] = {k: v.name for k, v in p.items()}
    manifest["raw_bytes"] = p["raw"].stat().st_size
    if public_probe is not None:
        manifest["public_probe"] = public_probe
    p["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
