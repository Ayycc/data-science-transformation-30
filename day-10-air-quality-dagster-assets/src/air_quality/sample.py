"""The committed sample (data/sample/, SYNTHETIC): registry + all on-time readings of 2026-09-01."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import FULL
from .generate import _frame, simulate, station_registry

SAMPLE_DAY = 0


def sample_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    sim = simulate(FULL)
    present = ~sim["missing"] & ~sim["lost"] & ~sim["late"]
    day_mask = (sim["day"] == SAMPLE_DAY)[:, None]
    readings = _frame(sim, present & day_mask, sim["landed"]).sort_values(["ts", "station_id"]).reset_index(drop=True)
    readings["landed_at"] = readings["landed_at"].dt.floor("s")
    return station_registry(sim), readings


def write_sample(out_dir: Path) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stations, readings = sample_frames()
    stations.to_csv(out_dir / "stations.csv", index=False)
    readings.to_csv(out_dir / "readings_2026-09-01.csv", index=False, float_format="%.1f")
    (out_dir / "README.md").write_text(
        "SYNTHETIC sample written by `scripts/generate_data.py` (seed 20261020).\n\n"
        "- `stations.csv`: the 300-station registry with known calibration offsets.\n"
        "- `readings_2026-09-01.csv`: every on-time reading of the first partition, before duplicates are added "
        "(hourly PM2.5 ug/m3, NO2 ppb, O3 ppb).\n"
    )
    return {p.name: p.stat().st_size for p in sorted(out_dir.iterdir())}


__all__ = ["sample_frames", "write_sample"]
