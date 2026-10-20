"""Paths, partitions and thresholds shared by the generator, the assets and the evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dagster import DailyPartitionsDefinition

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The month that is simulated and partitioned: 30 daily partitions, 2026-09-01 .. 2026-09-30.
START_DATE = "2026-09-01"
END_DATE_EXCLUSIVE = "2026-10-01"
N_DAYS = 30

daily_partitions = DailyPartitionsDefinition(start_date=START_DATE, end_date=END_DATE_EXCLUSIVE)

POLLUTANTS = ("pm25", "no2", "o3")
UNITS = {"pm25": "ug/m3", "no2": "ppb", "o3": "ppb"}
# Physically plausible bounds for a single hourly reading (anything outside is a sensor/unit fault).
PHYSICAL_BOUNDS = {"pm25": (0.0, 1000.0), "no2": (0.0, 2000.0), "o3": (0.0, 500.0)}
# Small negative readings are instrument noise around zero and are clipped to 0.
NEGATIVE_NOISE_FLOOR = -3.0

# Asset-check thresholds.
MIN_RAW_COMPLETENESS = 0.85  # share of registry station-hours present in raw_readings
MAX_SYSTEMIC_FAULT_RATE = 0.05  # share of raw values rejected as out of physical range
MIN_AQI_COVERAGE = 0.75  # share of registry stations with a valid PM2.5 sub-index
FRESHNESS_MAX_MEDIAN_LATENCY_MIN = 90.0  # median ingest latency (landed_at - ts)
ALERT_AQI = 100  # AQI above this (Unhealthy for Sensitive Groups or worse) raises an alert
REGION_ADVISORY_SHARE = 0.20  # share of a region's valid stations in alert that raises a regional advisory

# Drift detection (sensor_drift asset).
DRIFT_WINDOW_DAYS = 10
DRIFT_MIN_DAYS = 7
DRIFT_MIN_SLOPE = 0.5  # ug/m3 per day of PM2.5 bias growth
DRIFT_MIN_TSTAT = 6.0


def data_dir() -> Path:
    return Path(os.environ.get("AQ_DATA_DIR", PROJECT_ROOT / "data")).resolve()


def results_dir() -> Path:
    return Path(os.environ.get("AQ_RESULTS_DIR", PROJECT_ROOT / "results")).resolve()


@dataclass(frozen=True)
class Preset:
    """Generator scale and the planted faults. Day offsets are 0-based from START_DATE."""

    name: str
    n_stations: int = 300
    n_regions: int = 10
    n_days: int = N_DAYS
    seed: int = 20261020
    # Defects that hit every partition.
    random_gap_rate: float = 0.02
    duplicate_rate: float = 0.005
    spike_rate: float = 0.001
    outage_station_share: float = 0.03
    flatline_episodes_per_station: float = 0.5
    calibration_share: float = 0.10
    drift_share: float = 0.05
    # Planted incidents.
    heat_days: tuple[int, ...] = (2, 3, 4)
    heat_regions: tuple[int, ...] = (0, 1, 6)
    smoke_days: tuple[int, ...] = (18, 19, 20)
    smoke_regions: tuple[int, ...] = (2, 3)
    inversion_day: int | None = 7
    inversion_region: int = 5
    backhaul_outage_day: int | None = 11  # 2026-09-12: 60% of stations silent 04:00-20:00 on time
    backhaul_outage_region_share: float = 0.6
    backhaul_recovered_share: float = 0.85  # share of buffered readings that land late
    unit_bug_day: int | None = 23  # 2026-09-24: NO2 reported in ppt (x1000) by 45% of stations
    unit_bug_station_share: float = 0.45
    late_days: tuple[int, ...] = (5, 19, 26)  # evening readings of some stations land 2 days late
    late_station_share: float = 0.15
    late_first_hour: int = 17
    extra: dict = field(default_factory=dict)


FULL = Preset(name="full")
# Tiny preset for tests: 20 stations, 30 days, same incident calendar.
TINY = Preset(
    name="tiny",
    n_stations=20,
    n_regions=4,
    heat_regions=(0,),
    smoke_regions=(1,),
    inversion_region=2,
    late_first_hour=20,
)
PRESETS = {"full": FULL, "tiny": TINY}
