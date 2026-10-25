from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bikeshare import config as C


@pytest.fixture
def toy_stations() -> pd.DataFrame:
    """12 stations on a small grid around downtown Washington (SYNTHETIC fixture)."""
    rng = np.random.default_rng(0)
    lat = 38.90 + np.repeat(np.arange(4), 3) * 0.006
    lon = -77.04 + np.tile(np.arange(3), 4) * 0.008
    return pd.DataFrame(
        {
            "station_id": [f"S{i:02d}" for i in range(12)],
            "name": [f"Toy {i}" + (" Metro" if i == 4 else "") for i in range(12)],
            "lat": lat,
            "lon": lon,
            "activity": rng.integers(200, 2000, 12),
            "activity_per_day": rng.uniform(5, 60, 12),
            "is_metro": [i == 4 for i in range(12)],
            "elevation_m": rng.uniform(5, 90, 12).round(1),
            "capacity": 15,
        }
    )


@pytest.fixture
def sample_stations() -> pd.DataFrame:
    return pd.read_csv(C.DATA_SAMPLE / "stations.csv", dtype={"station_id": str})
