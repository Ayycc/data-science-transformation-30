"""Run configuration. One dataclass drives the full run, the out-of-core demo and the fast test fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field

SEED = 20261101

# Synthetic study region: a 4 x 6 degree box shaped like the US Corn Belt (SYNTHETIC climate, not real data).
LAT_MIN, LAT_MAX = 40.0, 44.0
LON_MIN, LON_MAX = -96.0, -90.0

# Drought years and their intensity (0-1). Each drought gets a seeded spatial footprint.
DROUGHT_YEARS: dict[int, float] = {2003: 0.5, 2006: 0.45, 2012: 1.0, 2017: 0.55, 2023: 0.75}

# Agronomy constants (maize-like crop).
GDD_BASE_C = 10.0
GDD_CAP_C = 30.0
HEAT_THRESHOLD_C = 32.0

# Climate scenario.
SCENARIO_DT_C = 2.0
SCENARIO_PRECIP_FACTOR = 0.8


@dataclass(frozen=True)
class CubeSpec:
    """Shape and storage chunking of one synthetic cube."""

    n_lat: int
    n_lon: int
    years: tuple[int, ...]
    chunk_time: int = 365
    chunk_lat: int = 40
    chunk_lon: int = 60

    @property
    def res_lat(self) -> float:
        return (LAT_MAX - LAT_MIN) / self.n_lat

    @property
    def res_lon(self) -> float:
        return (LON_MAX - LON_MIN) / self.n_lon


@dataclass(frozen=True)
class RunConfig:
    """Everything `make run` needs. `RunConfig.small()` is used by the tests."""

    cube: CubeSpec = field(default_factory=lambda: CubeSpec(80, 120, tuple(range(2001, 2025))))
    big_cube: CubeSpec = field(default_factory=lambda: CubeSpec(160, 240, tuple(range(2001, 2025)), 730, 40, 60))
    n_fields: int = 300
    block_deg: float = 1.0
    n_folds: int = 5
    big_mem_cap_gb: float = 1.0
    num_workers: int = 4
    seed: int = SEED

    @classmethod
    def small(cls) -> RunConfig:
        return cls(
            cube=CubeSpec(16, 24, tuple(range(2007, 2015)), 365, 8, 12),
            big_cube=CubeSpec(16, 24, tuple(range(2011, 2014)), 365, 8, 12),
            n_fields=40,
            block_deg=2.0,
            n_folds=3,
            big_mem_cap_gb=2.0,
            num_workers=2,
        )
