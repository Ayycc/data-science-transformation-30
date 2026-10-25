"""Project-wide constants: paths, study window, fleet parameters and seeds."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_SAMPLE = ROOT / "data" / "sample"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MAPS = RESULTS / "maps"

SEED = 20261025
TZ = "America/New_York"

# Public sources (Capital Bikeshare, Washington DC). Monthly trip archives on S3.
CABI_MONTHS = ("202408", "202409")
CABI_URL = "https://s3.amazonaws.com/capitalbikeshare-data/{month}-capitalbikeshare-tripdata.zip"
GBFS_STATION_INFO_URL = "https://gbfs.lyft.com/gbfs/2.3/dca/en/station_information.json"
# NOAA Integrated Surface Database, Reagan National Airport (DCA), hourly, UTC timestamps.
ISD_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/2024/72405013743.csv"
# AWS Terrain Tiles (Mapzen terrarium encoding), a public DEM mosaic.
TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TERRAIN_ZOOM = 12

# Study window (local time). Train on 46 days, test on two full weeks (Mon 16 Sep to Sun 29 Sep 2024).
TRAIN_START = pd.Timestamp("2024-08-01")
VALID_START = pd.Timestamp("2024-09-09")  # last training week used for early stopping
TEST_START = pd.Timestamp("2024-09-16")
TEST_END = pd.Timestamp("2024-09-30")  # exclusive
HOLIDAYS = (pd.Timestamp("2024-09-02"),)  # Labor Day

H3_RES = 8  # ~0.74 km^2 hexagons, edge ~0.46 km
MIN_STATION_TRIPS = 60  # starts + ends in the training window to keep a station


@dataclass(frozen=True)
class FleetConfig:
    """Rebalancing operation. Chosen before looking at test results; see README."""

    n_trucks: int = 8
    truck_capacity: int = 20  # bikes per truck
    speed_kmh: float = 18.0  # urban van speed
    circuity: float = 1.35  # road distance / straight-line distance
    service_s: int = 120  # parking + unlocking per stop
    per_bike_s: int = 20  # handling time per bike moved
    buffer: int = 2  # bikes / docks kept free of the forecast path
    min_move: int = 3  # smaller corrections are not worth a stop
    max_candidates: int = 150  # stations offered to the solver per shift
    penalty_per_bike_s: int = 900  # objective value of moving one bike (seconds of driving)
    solver_seconds: float = 3.0
    shifts: tuple[tuple[str, int, int, int], ...] = field(
        default=(
            # name, shift start hour, shift length (min), forecast horizon end hour
            ("overnight", 0, 300, 14),
            ("midday", 10, 240, 24),
        )
    )


FLEET = FleetConfig()
