"""Data acquisition: public Capital Bikeshare trips + NOAA weather, with a seeded SYNTHETIC fallback.

Every network call is wrapped: if a source is unreachable (proxy 403, timeout, DNS), the pipeline
records why and continues on the committed sample / the synthetic generator.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

USER_AGENT = "data-science-transformation-30/day-15 (research; contact via GitHub)"


def fetch(url: str, dest: Path | None = None, timeout: float = 60.0) -> tuple[bytes | None, str]:
    """Download ``url`` (cached at ``dest`` if given). Returns (content or None, status string)."""
    if dest is not None and dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes(), "cached"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
            status = f"reachable ({resp.status})"
    except Exception as exc:  # noqa: BLE001 - any failure means "use the fallback"
        return None, f"unreachable ({type(exc).__name__}: {str(exc)[:120]})"
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    return content, status


# --------------------------------------------------------------------------------------------- trips
TRIP_COLS = ["ride_id", "started_at", "ended_at", "start_station_id", "end_station_id", "start_station_name", "start_lat", "start_lng"]


def parse_cabi_zip(content: bytes) -> pd.DataFrame:
    """Parse one monthly Capital Bikeshare zip into a tidy trip table (docked trips only)."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv") and "MACOSX" not in n)
        raw = pd.read_csv(zf.open(name), usecols=TRIP_COLS, dtype={"start_station_id": "string", "end_station_id": "string"})
    raw = raw.dropna(subset=["start_station_id", "end_station_id"])
    out = pd.DataFrame(
        {
            "ride_id": raw["ride_id"].astype(str),
            "start_time": pd.to_datetime(raw["started_at"], format="ISO8601").dt.floor("s"),
            "end_time": pd.to_datetime(raw["ended_at"], format="ISO8601").dt.floor("s"),
            "start_id": raw["start_station_id"].astype(str).str.replace(r"\.0$", "", regex=True),
            "end_id": raw["end_station_id"].astype(str).str.replace(r"\.0$", "", regex=True),
            "start_name": raw["start_station_name"].astype(str),
            "start_lat": raw["start_lat"].astype(float),
            "start_lng": raw["start_lng"].astype(float),
        }
    )
    return out


def clean_trips(trips: pd.DataFrame, start: pd.Timestamp = C.TRAIN_START, end: pd.Timestamp = C.TEST_END) -> pd.DataFrame:
    dur = (trips["end_time"] - trips["start_time"]).dt.total_seconds()
    keep = (dur >= 60) & (dur <= 3 * 3600) & (trips["start_time"] >= start) & (trips["start_time"] < end)
    return trips.loc[keep].sort_values("start_time").reset_index(drop=True)


def stations_from_trips(trips: pd.DataFrame, train_end: pd.Timestamp = C.TEST_START, min_trips: int = C.MIN_STATION_TRIPS) -> pd.DataFrame:
    """Station table from trip records: median start coordinates, most common name, training activity."""
    tr = trips[trips["start_time"] < train_end]
    coords = tr.groupby("start_id").agg(
        lat=("start_lat", "median"), lon=("start_lng", "median"), name=("start_name", lambda s: s.mode().iat[0])
    )
    starts = tr["start_id"].value_counts().rename("n_starts")
    ends = tr["end_id"].value_counts().rename("n_ends")
    st = coords.join(starts, how="left").join(ends, how="left").fillna({"n_starts": 0, "n_ends": 0})
    st["activity"] = st["n_starts"] + st["n_ends"]
    st = st[st["activity"] >= min_trips].copy()
    st.index.name = "station_id"
    st = st.reset_index()
    st["is_metro"] = st["name"].str.contains(r"\bMetro\b(?! Park Dr)", case=False, regex=True)
    return st.sort_values("station_id").reset_index(drop=True)


def estimate_capacity(activity_per_day: pd.Series) -> pd.Series:
    """Dock-count proxy used when GBFS station_information is unreachable.

    Capital Bikeshare docks hold 11-39 bikes and busier stations get more docks; this monotone
    rule maps mean daily activity (starts + ends) onto that range.
    """
    return np.clip(np.round(9 + 1.5 * np.sqrt(activity_per_day.to_numpy(float))), 11, 39).astype(int)


def fetch_capacity() -> tuple[dict[str, int] | None, str]:
    content, status = fetch(C.GBFS_STATION_INFO_URL, timeout=15)
    if content is None:
        return None, status
    try:
        info = json.loads(content)["data"]["stations"]
        return {str(s.get("short_name", s["station_id"])): int(s["capacity"]) for s in info if s.get("capacity")}, status
    except Exception as exc:  # noqa: BLE001
        return None, f"unparseable ({exc})"


# --------------------------------------------------------------------------------------------- weather
def parse_isd(content: bytes) -> pd.DataFrame:
    """NOAA ISD hourly (FM-15 METAR) -> local hourly temperature (C) and precipitation (mm)."""
    d = pd.read_csv(io.BytesIO(content), usecols=["DATE", "REPORT_TYPE", "TMP", "AA1"], low_memory=False)
    d = d[d["REPORT_TYPE"].str.strip() == "FM-15"].copy()
    t = pd.to_datetime(d["DATE"]).dt.tz_localize("UTC").dt.tz_convert(C.TZ).dt.tz_localize(None)
    tmp = d["TMP"].str.split(",", expand=True)
    temp = pd.to_numeric(tmp[0], errors="coerce").where(lambda x: x.abs() < 900) / 10.0
    aa = d["AA1"].fillna("01,0000,9,9").str.split(",", expand=True)
    precip = pd.to_numeric(aa[1], errors="coerce").where(lambda x: x < 9000) / 10.0
    precip = precip.where(pd.to_numeric(aa[0], errors="coerce") == 1, 0.0)
    w = pd.DataFrame({"time": (t + pd.Timedelta(minutes=8)).dt.floor("h"), "temp_c": temp.to_numpy(), "precip_mm": precip.to_numpy()})
    w = w.groupby("time").agg(temp_c=("temp_c", "mean"), precip_mm=("precip_mm", "max"))
    return w


def synthetic_weather(start: pd.Timestamp, end: pd.Timestamp, seed: int = C.SEED) -> pd.DataFrame:
    """SYNTHETIC late-summer weather for DC: diurnal temperature cycle plus random rain spells."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h", inclusive="left")
    days = (idx - start).days.to_numpy()
    daily_mean = 27 - 6 * days / max(days.max(), 1) + np.repeat(rng.normal(0, 2, days.max() + 1), 1)[days]
    temp = daily_mean + 5 * np.sin(2 * np.pi * (idx.hour.to_numpy() - 9) / 24)
    rain = np.zeros(len(idx))
    for s in rng.choice(len(idx), size=max(1, len(idx) // 70), replace=False):
        rain[s : s + rng.integers(1, 6)] = rng.gamma(1.2, 1.5)
    return pd.DataFrame({"temp_c": np.round(temp, 1), "precip_mm": np.round(rain, 1)}, index=pd.Index(idx, name="time"))


def complete_weather(w: pd.DataFrame, start: pd.Timestamp = C.TRAIN_START, end: pd.Timestamp = C.TEST_END) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="h", inclusive="left", name="time")
    w = w.reindex(idx)
    w["temp_c"] = w["temp_c"].interpolate(limit_direction="both")
    w["precip_mm"] = w["precip_mm"].fillna(0.0)
    return w


# --------------------------------------------------------------------------------------------- orchestration
@dataclass
class Acquired:
    trips: pd.DataFrame
    stations: pd.DataFrame
    weather: pd.DataFrame
    provenance: dict = field(default_factory=dict)


def acquire(force_synthetic: bool = False) -> Acquired:
    """Real Capital Bikeshare data if reachable, else SYNTHETIC trips over the committed station layout."""
    from . import synthetic

    prov: dict = {"trips": {}, "weather": {}, "capacity": {}}
    frames: list[pd.DataFrame] = []
    if not force_synthetic:
        for month in C.CABI_MONTHS:
            url = C.CABI_URL.format(month=month)
            content, status = fetch(url, C.DATA_RAW / f"{month}-capitalbikeshare-tripdata.zip", timeout=180)
            prov["trips"][url] = status
            if content is None:
                frames = []
                break
            frames.append(parse_cabi_zip(content))

    # weather
    content, status = fetch(C.ISD_URL, C.DATA_RAW / "isd_72405013743_2024.csv", timeout=60)
    prov["weather"]["url"] = C.ISD_URL
    prov["weather"]["status"] = status
    if content is not None:
        weather = complete_weather(parse_isd(content))
        prov["weather"]["source"] = "NOAA ISD hourly, Reagan National Airport (public)"
    elif (C.DATA_SAMPLE / "weather_hourly.csv").exists():
        weather = complete_weather(pd.read_csv(C.DATA_SAMPLE / "weather_hourly.csv", parse_dates=["time"], index_col="time"))
        prov["weather"]["source"] = "committed sample of NOAA ISD hourly (public)"
    else:
        weather = synthetic_weather(C.TRAIN_START, C.TEST_END)
        prov["weather"]["source"] = "SYNTHETIC weather generator"

    if frames:
        trips = clean_trips(pd.concat(frames, ignore_index=True))
        stations = stations_from_trips(trips)
        trips = trips[trips["start_id"].isin(stations["station_id"]) & trips["end_id"].isin(stations["station_id"])]
        trips = trips[["ride_id", "start_time", "end_time", "start_id", "end_id"]].reset_index(drop=True)
        n_days = (C.TEST_START - C.TRAIN_START).days
        stations["activity_per_day"] = stations["activity"] / n_days
        prov["trips"]["source"] = "public Capital Bikeshare system data (docked trips, Aug-Sep 2024)"
        prov["trips"]["synthetic"] = False
    else:
        stations = pd.read_csv(C.DATA_SAMPLE / "stations.csv", dtype={"station_id": str})
        trips = synthetic.generate_trips(stations, weather, C.TRAIN_START, C.TEST_END, seed=C.SEED)
        prov["trips"]["source"] = "SYNTHETIC trips (seeded generator) over the committed real Capital Bikeshare station layout"
        prov["trips"]["synthetic"] = True

    caps, cap_status = fetch_capacity()
    prov["capacity"]["status"] = cap_status
    est = estimate_capacity(stations["activity_per_day"])
    if caps:
        real = stations["station_id"].map(caps)
        stations["capacity"] = real.fillna(pd.Series(est, index=stations.index)).astype(int)
        stations["capacity_source"] = np.where(real.notna(), "gbfs", "estimated")
    elif "capacity" in stations and not frames:
        pass  # committed sample already carries capacities
    else:
        stations["capacity"] = est
        stations["capacity_source"] = "estimated"
    prov["capacity"]["source"] = "GBFS station_information" if caps else "activity-based estimate (GBFS unreachable)"
    prov["n_trips"] = int(len(trips))
    prov["n_stations"] = int(len(stations))
    return Acquired(trips=trips, stations=stations, weather=weather, provenance=prov)


def write_samples(trips: pd.DataFrame, stations: pd.DataFrame, weather: pd.DataFrame) -> None:
    """Refresh data/sample/ (<= 1 MB in total) from the real data so the project runs offline."""
    C.DATA_SAMPLE.mkdir(parents=True, exist_ok=True)
    cols = ["station_id", "name", "lat", "lon", "n_starts", "n_ends", "activity", "activity_per_day", "is_metro", "elevation_m"]
    stations[[c for c in cols if c in stations]].round(6).to_csv(C.DATA_SAMPLE / "stations.csv", index=False)
    weather.round(2).reset_index().to_csv(C.DATA_SAMPLE / "weather_hourly.csv", index=False)
    # one weekday of real trips (Tue 17 Sep 2024) for tests and the quick path
    day = trips[(trips["start_time"] >= "2024-09-17") & (trips["start_time"] < "2024-09-18")]
    day = day.drop(columns="ride_id")
    day.to_csv(C.DATA_SAMPLE / "trips_2024-09-17.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
