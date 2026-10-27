"""Data acquisition: NYC TLC trip records if reachable, else a seeded SYNTHETIC city.

Both paths produce the same three event tables:

- ``trips``  : one row per fulfilled ride request (the label is ``duration_min``)
- ``offers`` : every dispatch offer sent to a driver and whether it was accepted
- ``probes`` : GPS speed observations (km/h) per zone, as drivers' phones would emit them

Every row carries the timestamp at which the event became *known*: offers at
offer time, probes at ping time, trip labels at drop-off time. The feature
pipeline only ever aggregates over those timestamps.
"""

from __future__ import annotations

import io
import json
import urllib.request
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths, SimConfig

TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"


# --------------------------------------------------------------------------------------
# Public path
# --------------------------------------------------------------------------------------
def try_download_tlc(url: str = TLC_URL, timeout: float = 15.0) -> tuple[pd.DataFrame | None, str]:
    """Try to fetch one month of NYC TLC yellow-taxi trips. Returns (frame or None, status)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
            payload = resp.read()
        return pd.read_parquet(io.BytesIO(payload)), "ok"
    except Exception as exc:  # network errors, proxy 403, parse errors
        return None, f"{type(exc).__name__}: {exc}"


def tlc_to_tables(tlc: pd.DataFrame, seed: int, n_drivers: int = 400, max_rows: int = 200_000) -> dict[str, pd.DataFrame]:
    """Map TLC yellow-taxi columns onto the trips/offers/probes schema.

    TLC records have no driver ids, offers or GPS pings, so those are derived:
    a seeded hash assigns drivers, each trip yields one accepted offer plus
    Bernoulli rejections, and each completed trip emits one probe (its average
    speed) in its pickup zone at drop-off time.
    """
    rng = np.random.default_rng(seed)
    df = tlc.rename(columns={"tpep_pickup_datetime": "request_ts", "tpep_dropoff_datetime": "dropoff_ts"})
    df = df[["request_ts", "dropoff_ts", "PULocationID", "DOLocationID", "trip_distance"]].dropna()
    df["request_ts"] = pd.to_datetime(df["request_ts"], utc=True)
    df["dropoff_ts"] = pd.to_datetime(df["dropoff_ts"], utc=True)
    df["duration_min"] = (df["dropoff_ts"] - df["request_ts"]).dt.total_seconds() / 60.0
    df["distance_km"] = df["trip_distance"] * 1.609344
    df = df[(df.duration_min.between(1, 120)) & (df.distance_km.between(0.3, 60))]
    df = df.sort_values("request_ts").head(max_rows).reset_index(drop=True)
    trips = pd.DataFrame(
        {
            "trip_id": np.arange(len(df), dtype=np.int64),
            "request_ts": df["request_ts"],
            "dropoff_ts": df["dropoff_ts"],
            "zone_id": df["PULocationID"].astype(np.int64),
            "dropoff_zone_id": df["DOLocationID"].astype(np.int64),
            "driver_id": rng.integers(0, n_drivers, len(df)).astype(np.int64),
            "distance_km": df["distance_km"].astype(float),
            "duration_min": df["duration_min"].astype(float),
        }
    )
    trips["free_flow_min"] = trips["distance_km"] / 30.0 * 60.0
    n_rej = rng.poisson(0.4, len(trips))
    rej = trips.loc[trips.index.repeat(n_rej)]
    offers = pd.concat(
        [
            pd.DataFrame({"driver_id": trips.driver_id, "offer_ts": trips.request_ts, "accepted": 1}),
            pd.DataFrame(
                {
                    "driver_id": rng.integers(0, n_drivers, len(rej)),
                    "offer_ts": rej.request_ts.to_numpy(),
                    "accepted": 0,
                }
            ),
        ],
        ignore_index=True,
    ).sort_values("offer_ts", kind="stable")
    probes = pd.DataFrame(
        {
            "zone_id": trips.zone_id,
            "probe_ts": trips.dropoff_ts,
            "speed_kmh": trips.distance_km / (trips.duration_min / 60.0),
        }
    ).sort_values("probe_ts", kind="stable")
    return _to_seconds({"trips": trips, "offers": offers.reset_index(drop=True), "probes": probes.reset_index(drop=True)})


# --------------------------------------------------------------------------------------
# SYNTHETIC path
# --------------------------------------------------------------------------------------
def _daily_profile(minutes_of_day: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    """Congestion shape in [0, 1]: morning and evening peaks on weekdays, a flat midday hump at weekends."""
    h = minutes_of_day / 60.0
    am = np.exp(-0.5 * ((h - 8.3) / 1.1) ** 2)
    pm = np.exp(-0.5 * ((h - 17.8) / 1.4) ** 2)
    mid = np.exp(-0.5 * ((h - 13.5) / 3.0) ** 2)
    weekday = 0.85 * am + 1.0 * pm + 0.35 * mid
    wkend = 0.55 * mid + 0.25 * np.exp(-0.5 * ((h - 22.5) / 1.5) ** 2)
    return np.where(weekend, wkend, weekday)


def _demand_profile(minutes_of_day: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    h = minutes_of_day / 60.0
    wd = 0.25 + 1.0 * np.exp(-0.5 * ((h - 8.5) / 1.3) ** 2) + 1.1 * np.exp(-0.5 * ((h - 18.0) / 1.8) ** 2) + 0.45 * np.exp(-0.5 * ((h - 13) / 2.5) ** 2)
    we = 0.3 + 0.6 * np.exp(-0.5 * ((h - 14) / 3.5) ** 2) + 0.9 * np.exp(-0.5 * ((h - 23.0) / 2.0) ** 2)
    night = 0.15 * (h < 5)
    return np.where(weekend, we, wd) - night


def simulate_city(cfg: SimConfig) -> dict[str, pd.DataFrame]:
    """Seeded SYNTHETIC city: zone traffic with incidents, demand, dispatch offers and trips."""
    rng = np.random.default_rng(cfg.seed)
    nz, g = cfg.n_zones, cfg.grid
    tick = pd.Timedelta(minutes=cfg.tick_min)
    n_ticks = cfg.n_days * 24 * 60 // cfg.tick_min
    t0 = pd.Timestamp(cfg.start, tz="UTC")
    tick_ts = pd.DatetimeIndex(t0 + np.arange(n_ticks) * tick)
    mod = (np.arange(n_ticks) * cfg.tick_min) % (24 * 60)
    weekend = np.asarray(tick_ts.dayofweek) >= 5

    # zones: 6x6 grid, centre is the CBD
    zx, zy = np.meshgrid(np.arange(g), np.arange(g))
    zx, zy = zx.ravel(), zy.ravel()
    centre = (g - 1) / 2.0
    cbd = np.exp(-(((zx - centre) ** 2 + (zy - centre) ** 2) / (2 * 1.6**2)))  # 1 at centre
    free_flow = 42.0 - 16.0 * cbd + rng.normal(0, 2.0, nz)  # km/h

    # true congestion c(z, t) = profile * weight + AR(1) + incidents
    prof = _daily_profile(mod, weekend)  # (T,)
    weight = 0.25 + 0.4 * cbd + rng.uniform(-0.05, 0.05, nz)
    ar = np.zeros((nz, n_ticks))
    eps = rng.normal(0, 0.03, (nz, n_ticks))
    for t in range(1, n_ticks):
        ar[:, t] = 0.97 * ar[:, t - 1] + eps[:, t]
    incident = np.zeros((nz, n_ticks))
    n_inc = rng.poisson(0.8 * cfg.n_days, nz)
    for z in range(nz):
        for _ in range(n_inc[z]):
            s = rng.integers(0, n_ticks)
            d = rng.integers(4, 25)  # 20-120 minutes
            sev = rng.uniform(0.25, 0.5)
            ramp = np.minimum(1.0, (np.arange(d) + 1) / 2.0)  # reaches full severity within 10 minutes
            e = min(n_ticks, s + d)
            incident[z, s:e] = np.maximum(incident[z, s:e], sev * ramp[: e - s])
    cong = np.clip(prof[None, :] * weight[:, None] + ar + incident, 0.0, 0.85)
    speed = free_flow[:, None] * (1.0 - cong)  # (Z, T) km/h

    # probes: phones in each zone ping their speed
    activity = _demand_profile(mod, weekend).clip(0.05)
    lam_p = (2.0 + 6.0 * cbd[:, None]) * activity[None, :] * 1.5
    n_p = rng.poisson(lam_p)
    zi, ti = np.nonzero(n_p)
    reps = n_p[zi, ti]
    pz, pt = np.repeat(zi, reps), np.repeat(ti, reps)
    off_s = rng.uniform(0, cfg.tick_min * 60, len(pz))
    probes = pd.DataFrame(
        {
            "zone_id": pz.astype(np.int64),
            "probe_ts": tick_ts[pt] + pd.to_timedelta(off_s, unit="s"),
            "speed_kmh": speed[pz, pt] * rng.lognormal(0, 0.18, len(pz)),
        }
    )

    # requests
    zone_demand = 0.4 + 1.6 * cbd + rng.uniform(0, 0.3, nz)
    per_tick = cfg.requests_per_day / (24 * 60 / cfg.tick_min)
    dem = _demand_profile(mod, weekend).clip(0.05)
    lam_r = zone_demand[:, None] / zone_demand.sum() * per_tick * dem[None, :] / dem.mean()
    n_r = rng.poisson(lam_r)
    zi, ti = np.nonzero(n_r)
    reps = n_r[zi, ti]
    rz, rt = np.repeat(zi, reps), np.repeat(ti, reps)
    r_ts = tick_ts[rt] + pd.to_timedelta(rng.uniform(0, cfg.tick_min * 60, len(rz)), unit="s")
    n_req = len(rz)
    # drop-off zone: gravity towards the CBD, never too far
    attract = 0.5 + 2.0 * cbd
    dist_z = np.abs(zx[:, None] - zx[None, :]) + np.abs(zy[:, None] - zy[None, :])
    p_dz = attract[None, :] * np.exp(-0.45 * dist_z)
    p_dz /= p_dz.sum(1, keepdims=True)
    cum = p_dz.cumsum(1)
    dz = (rng.uniform(size=n_req)[:, None] > cum[rz]).sum(1).clip(0, nz - 1)
    # distance between random points in the two zones, with a road detour factor
    px = (zx[rz] + rng.uniform(0, 1, n_req)) * cfg.zone_km
    py = (zy[rz] + rng.uniform(0, 1, n_req)) * cfg.zone_km
    qx = (zx[dz] + rng.uniform(0, 1, n_req)) * cfg.zone_km
    qy = (zy[dz] + rng.uniform(0, 1, n_req)) * cfg.zone_km
    dist = (np.abs(px - qx) + np.abs(py - qy)) * rng.uniform(1.05, 1.35, n_req) + 0.4

    # drivers: acceptance propensity and a slowly drifting speed skill
    acc_base = rng.beta(7, 2.5, cfg.n_drivers)
    skill_walk = rng.normal(0, 0.08, cfg.n_drivers)[:, None] + np.cumsum(rng.normal(0, 0.025, (cfg.n_drivers, cfg.n_days)), axis=1)
    skill = np.exp(skill_walk)  # (D, days), multiplies speed

    # dispatch: offer up to k random drivers in turn; first acceptance wins
    k = cfg.candidates_per_request
    cand = rng.integers(0, cfg.n_drivers, (n_req, k))
    hour = (r_ts.hour.to_numpy()).astype(int)
    late = (hour >= 23) | (hour < 5)
    p_acc = acc_base[cand] * np.where(late, 0.85, 1.0)[:, None]
    acc = rng.uniform(size=(n_req, k)) < p_acc
    first = np.where(acc.any(1), acc.argmax(1), -1)
    sent = np.arange(k)[None, :] <= np.where(first >= 0, first, k - 1)[:, None]
    oi, oj = np.nonzero(sent)
    offer_delay = pd.to_timedelta(oj * 20, unit="s")
    offers = pd.DataFrame(
        {
            "driver_id": cand[oi, oj].astype(np.int64),
            "offer_ts": r_ts[oi] + offer_delay,
            "accepted": acc[oi, oj].astype(np.int64),
        }
    ).sort_values("offer_ts", kind="stable")

    ok = first >= 0
    idx = np.nonzero(ok)[0]
    drv = cand[idx, first[idx]]
    day = ((rt[idx] * cfg.tick_min) // (24 * 60)).clip(0, cfg.n_days - 1)
    sk = skill[drv, day]
    # travel time uses the traffic that the trip will actually meet (future, unknown at request time)
    v0 = speed[rz[idx], rt[idx]]
    est = dist[idx] / v0 * 60.0
    t_mid1 = np.minimum(rt[idx] + (est * 0.25 / cfg.tick_min).astype(int), n_ticks - 1)
    t_mid2 = np.minimum(rt[idx] + (est * 0.75 / cfg.tick_min).astype(int), n_ticks - 1)
    v1 = speed[rz[idx], t_mid1]
    v2 = speed[dz[idx], t_mid2]
    v_eff = 2.0 / (1.0 / v1 + 1.0 / v2) * sk
    pickup_overhead = 1.0 + 3.0 * (1.0 - acc_base[drv]) + rng.gamma(2.0, 0.4, len(idx))
    duration = pickup_overhead + dist[idx] / v_eff * 60.0 * rng.lognormal(0, 0.08, len(idx))
    trips = pd.DataFrame(
        {
            "trip_id": np.arange(len(idx), dtype=np.int64),
            "request_ts": r_ts[idx],
            "zone_id": rz[idx].astype(np.int64),
            "dropoff_zone_id": dz[idx].astype(np.int64),
            "driver_id": drv.astype(np.int64),
            "distance_km": dist[idx],
            "duration_min": duration,
        }
    )
    trips["dropoff_ts"] = trips["request_ts"] + pd.to_timedelta(trips["duration_min"], unit="m")
    trips["free_flow_min"] = trips["distance_km"] / free_flow[trips["zone_id"].to_numpy()] * 60.0
    trips = trips.sort_values("request_ts", kind="stable").reset_index(drop=True)
    trips["trip_id"] = np.arange(len(trips), dtype=np.int64)
    probes = probes.sort_values("probe_ts", kind="stable").reset_index(drop=True)
    return _to_seconds({"trips": trips, "offers": offers.reset_index(drop=True), "probes": probes})


def _to_seconds(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Event clocks tick in whole seconds (as GPS and dispatch logs do); avoids ns->us casts downstream."""
    for df in tables.values():
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.floor("s").astype("datetime64[us, UTC]")
    return tables


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------
def acquire(cfg: SimConfig, paths: Paths, allow_network: bool = True) -> dict:
    """Write trips/offers/probes parquet under data/raw and a <=1 MB sample. Returns provenance."""
    paths.raw.mkdir(parents=True, exist_ok=True)
    tables, status = None, "network disabled"
    if allow_network:
        tlc, status = try_download_tlc()
        if tlc is not None:
            tables = tlc_to_tables(tlc, cfg.seed, cfg.n_drivers)
    source = "nyc_tlc_yellow_2024_01" if tables is not None else "synthetic"
    if tables is None:
        tables = simulate_city(cfg)
    for name, df in tables.items():
        df.to_parquet(paths.raw / f"{name}.parquet", index=False)
    write_sample(tables, paths.sample)
    prov = {
        "source": source,
        "tlc_url": TLC_URL,
        "tlc_status": status,
        "config": asdict(cfg),
        "rows": {k: int(len(v)) for k, v in tables.items()},
        "time_range": [str(tables["trips"].request_ts.min()), str(tables["trips"].request_ts.max())],
    }
    (paths.raw / "provenance.json").write_text(json.dumps(prov, indent=2))
    return prov


def write_sample(tables: dict[str, pd.DataFrame], out: Path, hours: int = 6) -> None:
    """Small committed sample: the first day's 06:00-12:00 window of each table (well under 1 MB)."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = tables["trips"].request_ts.min().normalize() + pd.Timedelta(hours=6)
    t1 = t0 + pd.Timedelta(hours=hours)
    cols = {"trips": "request_ts", "offers": "offer_ts", "probes": "probe_ts"}
    for name, col in cols.items():
        df = tables[name]
        part = df[(df[col] >= t0) & (df[col] < t1)]
        if name == "probes":
            part = part.iloc[::4]  # thin the GPS pings to stay small
        part.to_csv(out / f"{name}_sample.csv.gz", index=False, float_format="%.4f", compression="gzip")


def load_tables(paths: Paths) -> dict[str, pd.DataFrame]:
    return {n: pd.read_parquet(paths.raw / f"{n}.parquet") for n in ("trips", "offers", "probes")}
