"""Run configuration: scale, seeds and paths (one place, so tests can shrink it)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DAY_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SimConfig:
    """Scale of the SYNTHETIC city. Defaults are what `make run` uses."""

    seed: int = 20261027
    start: str = "2026-09-01"  # UTC, a Tuesday
    n_days: int = 28
    grid: int = 6  # grid x grid zones, 2 km apart
    zone_km: float = 2.0
    n_drivers: int = 400
    requests_per_day: float = 6000.0
    tick_min: int = 5
    candidates_per_request: int = 3
    train_days: int = 21  # days 1-21 train (first 2 used only as feature warm-up), 22-28 test
    warmup_days: int = 2

    @property
    def n_zones(self) -> int:
        return self.grid * self.grid


@dataclass(frozen=True)
class Paths:
    root: Path = DAY_DIR
    data: Path = field(default_factory=lambda: DAY_DIR / "data")
    results: Path = field(default_factory=lambda: DAY_DIR / "results")

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def offline(self) -> Path:
        return self.data / "offline"

    @property
    def store(self) -> Path:
        return self.data / "store"

    @property
    def sample(self) -> Path:
        return self.data / "sample"

    @property
    def figures(self) -> Path:
        return self.results / "figures"
