"""Constants shared by the generator, the backtest and the tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
SAMPLE_DIR = DATA_DIR / "sample"
RESULTS_DIR = ROOT / "results"

SEED = 42
SAMPLE_SEED = 43

# Public source probed before falling back to the synthetic generator (EIA-930 six-month balance file, no key).
EIA930_PROBE_URL = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2024_Jan_Jun.csv"

HORIZON = 24  # day-ahead: forecasts issued at 23:00 for hours 00:00..23:00 of the next day
LEVEL = 90  # nominal coverage of the prediction intervals, in percent
QUANTILES = (0.05, 0.5, 0.95)


@dataclass(frozen=True)
class Region:
    """Parameters of one SYNTHETIC balancing region."""

    name: str
    base_mw: float  # mean load at the comfort temperature
    temp_mean: float  # annual mean temperature, deg C
    temp_amp: float  # annual half-range, deg C
    diurnal_amp: float  # daily half-range, deg C
    heat_mw_per_deg: float  # heating load per degree below 15 C (fraction of base)
    cool_mw_per_deg: float  # cooling load per degree above 20 C (fraction of base)
    peak_shape: float  # weight of the evening peak relative to the daytime plateau


REGIONS: tuple[Region, ...] = (
    Region("NORTH", 6_000, 6.0, 14.0, 4.0, 0.022, 0.010, 1.2),
    Region("SOUTH", 9_000, 21.0, 8.0, 5.0, 0.008, 0.030, 0.9),
    Region("EAST", 11_000, 13.0, 12.0, 4.5, 0.012, 0.020, 1.0),
    Region("WEST", 7_000, 15.0, 6.0, 6.0, 0.006, 0.012, 1.1),
    Region("CENTRAL", 4_000, 11.0, 16.0, 6.0, 0.016, 0.022, 1.0),
)

# Regime shift: a data-centre cluster connects in CENTRAL, ramping in over two weeks.
REGIME_REGION = "CENTRAL"
REGIME_START = "2024-04-01"
REGIME_RAMP_DAYS = 14
REGIME_UPLIFT = 0.12  # +12% of base load, flat across hours


@dataclass(frozen=True)
class Scale:
    """How much data to generate and how many folds to backtest."""

    name: str
    start: str
    end: str  # exclusive
    regions: tuple[str, ...]
    n_folds: int
    fold_days: int
    cal_days: int  # conformal calibration window directly before each fold
    seed: int
    val_start: str  # first validation fold (used only to choose the conformal configuration)
    n_val_folds: int
    val_spacing_days: int


FULL = Scale(
    name="full",
    start="2021-01-01",
    end="2025-01-01",
    regions=tuple(r.name for r in REGIONS),
    n_folds=12,
    fold_days=28,
    cal_days=56,
    seed=SEED,
    val_start="2023-02-01",
    n_val_folds=6,
    val_spacing_days=56,
)

# Committed sample: 2 regions, 7 months. Small enough to commit (< 1 MB) and to backtest in tests.
SAMPLE = Scale(
    name="sample",
    start="2024-01-01",
    end="2024-08-01",
    regions=("NORTH", "CENTRAL"),
    n_folds=2,
    fold_days=7,
    cal_days=28,
    seed=SAMPLE_SEED,
    val_start="2024-07-04",
    n_val_folds=1,
    val_spacing_days=7,
)

SCALES = {"full": FULL, "sample": SAMPLE}

# Conformal configurations compared on the validation folds; the best one is used for the test backtest.
# (method, rolling window in days, ACI step size or None, Mondrian block of horizons in hours)
CONFORMAL_GRID: tuple[tuple[str, int, float | None, int], ...] = tuple(
    (m, w, g, b) for m in ("SplitCP", "CQR") for w in (28, 56) for g in (None, 0.05) for b in (24, 6)
)

# Temperature stress scenarios (stretch goal), applied to all regions in the test year.
STRESS_SCENARIOS: tuple[tuple[str, str, str, float], ...] = (
    ("heatwave_+4C", "2024-07-22", "2024-07-29", 4.0),
    ("cold_snap_-6C", "2024-02-05", "2024-02-12", -6.0),
)
