"""SYNTHETIC air-quality sensor network generator (seeded, fully deterministic).

Simulates hourly PM2.5 (ug/m3), NO2 (ppb) and O3 (ppb) for a network of stations grouped in regions,
then corrupts the measurements the way real networks do: calibration offsets, slow sensor drift,
station outages, random gaps, stuck loggers (flatlines), spikes, duplicate deliveries, late deliveries,
one network backhaul outage and one firmware unit bug. The noise-free-of-faults "truth" is written next
to the landing zone so the pipeline can be scored, but the pipeline itself never reads it.

Layout written under ``out_dir``:

- ``reference/stations.parquet``         station registry (incl. known calibration offsets)
- ``landing/<date>/batch_000.parquet``   readings that arrive on time
- ``late_arrivals/<date>/batch_001.parquet`` readings that arrive days later (delivered by the run script)
- ``truth/truth_hourly.parquet`` and ``truth/stations_truth.parquet``  ground truth for evaluation only
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import START_DATE, Preset


def _gauss(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return np.exp(-((x - mu) ** 2) / (2 * sd**2))


def _ar1_hourly(rng: np.random.Generator, n_days: int, n_regions: int, phi: float, sd: float) -> np.ndarray:
    """Daily AR(1) log-multiplier per region, linearly interpolated to hours."""
    z = np.zeros((n_days + 1, n_regions))
    z[0] = rng.normal(0, sd / np.sqrt(1 - phi**2), n_regions)
    for d in range(1, n_days + 1):
        z[d] = phi * z[d - 1] + rng.normal(0, sd, n_regions)
    t = np.arange(n_days * 24) / 24.0
    return np.stack([np.interp(t, np.arange(n_days + 1), z[:, r]) for r in range(n_regions)], axis=1)


def simulate(preset: Preset) -> dict:
    """Return all arrays/tables of one synthetic month (no I/O)."""
    rng = np.random.default_rng(preset.seed)
    S, R, D = preset.n_stations, preset.n_regions, preset.n_days
    H = D * 24
    hours = np.arange(H)
    hod = hours % 24
    day = hours // 24
    t0 = pd.Timestamp(START_DATE)
    ts = t0 + pd.to_timedelta(hours, unit="h")

    # ---- stations -------------------------------------------------------------------------
    region = rng.permutation(np.arange(S) % R)
    traffic = rng.random(S) < 0.25
    centers = np.stack([np.arange(R) // 5, np.arange(R) % 5], axis=1).astype(float)
    lat = 40.0 + centers[region, 0] * 0.6 + rng.normal(0, 0.12, S)
    lon = -3.0 + centers[region, 1] * 0.6 + rng.normal(0, 0.12, S)
    scale = rng.uniform(0.92, 1.08, S)
    static_pm = rng.normal(0, 1.5, S)
    static_no2 = rng.normal(0, 3.0, S)
    station_id = np.array([f"ST{i:03d}" for i in range(S)])

    # ---- regional fields (H x R) --------------------------------------------------------------
    base_pm = rng.uniform(6, 12, R)
    z_pm = _ar1_hourly(rng, D, R, 0.7, 0.22)
    pm_reg = base_pm[None, :] * np.exp(z_pm) * (1 + 0.2 * np.cos(2 * np.pi * (hod - 8) / 12))[:, None]
    smoke_amp = {0: 35.0, 1: 75.0, 2: 40.0}
    for k, d in enumerate(preset.smoke_days):
        shape = 0.7 + 0.6 * _gauss(hod, 20, 3)
        for r in preset.smoke_regions:
            pm_reg[day == d, r] += smoke_amp.get(k, 40.0) * shape[day == d]
    if preset.inversion_day is not None:
        m = day == preset.inversion_day
        pm_reg[m, preset.inversion_region] += 10 + 30 * _gauss(hod[m], 6, 3)

    z_no2 = _ar1_hourly(rng, D, R, 0.6, 0.25)
    traffic_shape = 0.6 + 0.6 * _gauss(hod, 8, 1.5) + 0.7 * _gauss(hod, 18, 1.5)
    no2_reg = 12.0 * np.exp(z_no2) * traffic_shape[:, None]
    if preset.inversion_day is not None:
        no2_reg[day == preset.inversion_day, preset.inversion_region] *= 1.8

    z_o3 = _ar1_hourly(rng, D, R, 0.7, 0.2)
    sun = np.clip(np.sin(np.pi * (hod - 8) / 12), 0, None)
    o3_reg = 25.0 * np.exp(0.5 * z_o3) + 20.0 * sun[:, None]
    for d in preset.heat_days:
        for r in preset.heat_regions:
            o3_reg[day == d, r] += 45.0 * sun[day == d]

    # ---- station truth (H x S) -----------------------------------------------------------------
    pm_t = pm_reg[:, region] * scale + static_pm
    pm_t = pm_t + rng.normal(0, 1, (H, S)) * 0.06 * pm_reg[:, region]
    no2_t = no2_reg[:, region] * scale * np.where(traffic, 1.8, 1.0) + static_no2
    no2_t = no2_t + rng.normal(0, 1, (H, S)) * 0.06 * no2_reg[:, region]
    o3_t = o3_reg[:, region] * scale * np.where(traffic, 0.85, 1.0)
    o3_t = o3_t + rng.normal(0, 1, (H, S)) * 0.06 * o3_reg[:, region]
    truth = {"pm25": np.clip(pm_t, 0, None), "no2": np.clip(no2_t, 0, None), "o3": np.clip(o3_t, 0, None)}

    # ---- measurement error: calibration offsets (known) and drift (unknown) --------------------
    order = rng.permutation(S)
    n_cal = int(round(preset.calibration_share * S))
    n_drift = max(1, int(round(preset.drift_share * S)))
    cal_idx, drift_idx = order[:n_cal], order[n_cal : n_cal + n_drift]
    cal_pm = np.zeros(S)
    cal_no2 = np.zeros(S)
    cal_pm[cal_idx] = rng.choice([-1, 1], n_cal) * rng.uniform(4, 10, n_cal)
    cal_no2[cal_idx] = rng.choice([-1, 1], n_cal) * rng.uniform(3, 8, n_cal)
    drift_rate = np.zeros(S)
    drift_start = np.full(S, -1)
    drift_rate[drift_idx] = np.where(rng.random(n_drift) < 2 / 3, 1, -1) * rng.uniform(0.8, 1.6, n_drift)
    drift_start[drift_idx] = rng.integers(2, max(3, min(15, D - 3)), n_drift)
    t_days = hours / 24.0
    drift = np.where(
        drift_start[None, :] >= 0, drift_rate[None, :] * np.clip(t_days[:, None] - drift_start[None, :], 0, None), 0.0
    )

    meas = {}
    for p in ("pm25", "no2", "o3"):
        noise = rng.normal(0, 1, (H, S)) * (0.8 + 0.05 * truth[p])
        meas[p] = truth[p] + noise
    meas["pm25"] = meas["pm25"] + cal_pm + drift
    meas["no2"] = meas["no2"] + cal_no2

    # ---- data faults ---------------------------------------------------------------------------
    missing = rng.random((H, S)) < preset.random_gap_rate
    n_out = int(round(preset.outage_station_share * S))
    outage_days: dict[str, list[int]] = {}
    for s in rng.choice(S, n_out, replace=False):
        start = int(rng.integers(0, max(1, D - 4)))
        length = int(rng.integers(1, 5))
        missing[(day >= start) & (day < start + length), s] = True
        outage_days[station_id[s]] = list(range(start, min(D, start + length)))
    n_flat = rng.poisson(preset.flatline_episodes_per_station, S)
    flat_mask = np.zeros((H, S), dtype=bool)
    for s in range(S):
        for _ in range(n_flat[s]):
            h0 = int(rng.integers(0, H - 13))
            ln = int(rng.integers(6, 13))
            for p in meas:
                meas[p][h0 : h0 + ln, s] = meas[p][h0, s]
            flat_mask[h0 : h0 + ln, s] = True
    spike_mask = {}
    for p in meas:
        m = rng.random((H, S)) < preset.spike_rate
        meas[p][m] = meas[p][m] * rng.uniform(8, 20, m.sum()) + 30
        spike_mask[p] = m
    unit_bug_stations = np.zeros(S, dtype=bool)
    if preset.unit_bug_day is not None:
        unit_bug_stations = rng.random(S) < preset.unit_bug_station_share
        m = (day == preset.unit_bug_day)[:, None] & unit_bug_stations[None, :]
        meas["no2"] = np.where(m, meas["no2"] * 1000.0, meas["no2"])
    for p in meas:
        meas[p] = np.round(meas[p], 1)

    # ---- delivery: on-time vs late ---------------------------------------------------------------
    late = np.zeros((H, S), dtype=bool)
    lost = np.zeros((H, S), dtype=bool)
    late_landed = np.full((H, S), np.datetime64("NaT"), dtype="datetime64[ns]")
    if preset.backhaul_outage_day is not None:
        d = preset.backhaul_outage_day
        n_r = int(round(preset.backhaul_outage_region_share * R))
        hit = np.isin(region, np.arange(n_r))
        window = (day == d) & (hod >= 4) & (hod <= 20)
        buffered = window[:, None] & hit[None, :] & ~missing
        recovered = rng.random((H, S)) < preset.backhaul_recovered_share
        late |= buffered & recovered
        lost |= buffered & ~recovered
        late_landed[buffered & recovered] = np.datetime64(t0 + pd.Timedelta(days=d + 2, hours=8))
    for d in preset.late_days:
        if d >= D:
            continue
        stations = rng.random(S) < preset.late_station_share
        m = ((day == d) & (hod >= preset.late_first_hour))[:, None] & stations[None, :] & ~missing
        late |= m
        late_landed[m] = np.datetime64(t0 + pd.Timedelta(days=d + 2, hours=6))

    latency_min = 60 + rng.exponential(20, (H, S))
    landed = ts.values[:, None] + (latency_min * 60e9).astype("timedelta64[ns]")

    return {
        "preset": preset,
        "ts": ts,
        "day": day,
        "station_id": station_id,
        "region": region,
        "traffic": traffic,
        "lat": lat,
        "lon": lon,
        "cal_pm": cal_pm,
        "cal_no2": cal_no2,
        "drift_rate": drift_rate,
        "drift_start": drift_start,
        "truth": truth,
        "meas": meas,
        "missing": missing,
        "late": late,
        "lost": lost,
        "landed": landed,
        "late_landed": late_landed,
        "flat_mask": flat_mask,
        "spike_mask": spike_mask,
        "unit_bug_stations": unit_bug_stations,
        "outage_days": outage_days,
        "rng": rng,
    }


def _frame(sim: dict, mask: np.ndarray, landed: np.ndarray) -> pd.DataFrame:
    hh, ss = np.nonzero(mask)
    return pd.DataFrame(
        {
            "station_id": sim["station_id"][ss],
            "ts": sim["ts"].values[hh],
            "pm25": sim["meas"]["pm25"][hh, ss],
            "no2": sim["meas"]["no2"][hh, ss],
            "o3": sim["meas"]["o3"][hh, ss],
            "landed_at": landed[hh, ss],
        }
    )


def station_registry(sim: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": sim["station_id"],
            "region": [f"R{r:02d}" for r in sim["region"]],
            "station_type": np.where(sim["traffic"], "traffic", "background"),
            "lat": np.round(sim["lat"], 5),
            "lon": np.round(sim["lon"], 5),
            "cal_offset_pm25": np.round(sim["cal_pm"], 2),
            "cal_offset_no2": np.round(sim["cal_no2"], 2),
        }
    )


def write_dataset(preset: Preset, out_dir: Path, write_truth: bool = True) -> dict:
    """Generate the month and write the landing zone, late arrivals, registry and truth."""
    out_dir = Path(out_dir)
    sim = simulate(preset)
    rng = sim["rng"]
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)
    station_registry(sim).to_parquet(out_dir / "reference" / "stations.parquet", index=False)

    present = ~sim["missing"] & ~sim["lost"]
    on_time = present & ~sim["late"]
    n_rows = {"on_time": 0, "late": 0, "duplicates": 0}
    dates = pd.date_range(START_DATE, periods=preset.n_days, freq="D")
    for d, date in enumerate(dates):
        dm = (sim["day"] == d)[:, None]
        df = _frame(sim, on_time & dm, sim["landed"])
        n_dup = int(round(preset.duplicate_rate * len(df)))
        if n_dup:
            dup = df.iloc[np.sort(rng.choice(len(df), n_dup, replace=False))].copy()
            dup["landed_at"] = dup["landed_at"] + pd.Timedelta(minutes=30)
            df = pd.concat([df, dup], ignore_index=True)
            n_rows["duplicates"] += n_dup
        df = df.sort_values(["ts", "station_id", "landed_at"], kind="stable").reset_index(drop=True)
        p = out_dir / "landing" / date.strftime("%Y-%m-%d")
        p.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p / "batch_000.parquet", index=False)
        n_rows["on_time"] += len(df)
        late_df = _frame(sim, sim["late"] & dm, sim["late_landed"])
        if len(late_df):
            lp = out_dir / "late_arrivals" / date.strftime("%Y-%m-%d")
            lp.mkdir(parents=True, exist_ok=True)
            late_df.sort_values(["ts", "station_id"]).to_parquet(lp / "batch_001.parquet", index=False)
            n_rows["late"] += len(late_df)

    summary = {
        "data_source": "SYNTHETIC (seeded generator src/air_quality/generate.py)",
        "preset": preset.name,
        "seed": preset.seed,
        "n_stations": preset.n_stations,
        "n_regions": preset.n_regions,
        "n_days": preset.n_days,
        "station_hours": int(preset.n_stations * preset.n_days * 24),
        "rows_on_time_incl_duplicates": n_rows["on_time"],
        "rows_late": n_rows["late"],
        "duplicate_rows": n_rows["duplicates"],
        "missing_station_hours": int(sim["missing"].sum()),
        "lost_station_hours": int(sim["lost"].sum()),
        "flatline_station_hours": int(sim["flat_mask"].sum()),
        "spike_values": int(sum(m.sum() for m in sim["spike_mask"].values())),
        "calibrated_stations": int((sim["cal_pm"] != 0).sum()),
        "drifting_stations": int((sim["drift_start"] >= 0).sum()),
        "outage_stations": len(sim["outage_days"]),
        "unit_bug_stations": int(sim["unit_bug_stations"].sum()) if preset.unit_bug_day is not None else 0,
        "late_partitions": sorted({str(p.name) for p in (out_dir / "late_arrivals").glob("*")})
        if (out_dir / "late_arrivals").exists()
        else [],
    }
    if write_truth:
        (out_dir / "truth").mkdir(parents=True, exist_ok=True)
        H, S = sim["missing"].shape
        hh, ss = np.meshgrid(np.arange(H), np.arange(S), indexing="ij")
        pd.DataFrame(
            {
                "station_id": sim["station_id"][ss.ravel()],
                "ts": sim["ts"].values[hh.ravel()],
                "pm25": sim["truth"]["pm25"].ravel(),
                "no2": sim["truth"]["no2"].ravel(),
                "o3": sim["truth"]["o3"].ravel(),
            }
        ).to_parquet(out_dir / "truth" / "truth_hourly.parquet", index=False)
        pd.DataFrame(
            {
                "station_id": sim["station_id"],
                "drift_rate": sim["drift_rate"],
                "drift_start_day": sim["drift_start"],
                "unit_bug": sim["unit_bug_stations"],
            }
        ).to_parquet(out_dir / "truth" / "stations_truth.parquet", index=False)
        (out_dir / "truth" / "generator_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
