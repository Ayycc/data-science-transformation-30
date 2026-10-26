"""Paths, seeds and policy thresholds shared across the pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_SAMPLE = ROOT / "data" / "sample"
RESULTS = ROOT / "results"
MLRUNS = ROOT / "mlruns"
MODEL_EXPORT = ROOT / "models" / "champion"

SEED = 20261026
MODEL_NAME = "claims-pure-premium"
EXPERIMENT = "day16-claims-severity"
TWEEDIE_POWER = 1.5

CAT_FEATURES = ["Area", "VehBrand", "VehGas", "Region"]
NUM_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "PurePremium"  # ClaimAmount / Exposure (EUR per policy-year)
WEIGHT = "Exposure"

AREAS = ["A", "B", "C", "D", "E", "F"]
BRANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"]
GAS = ["Regular", "Diesel"]
REGIONS = ["R11", "R24", "R31", "R52", "R53", "R72", "R82", "R91", "R93", "R94"]


def tracking_uri(root: Path | None = None) -> str:
    """Local SQLite tracking + registry store (no server). Overridable via MLFLOW_TRACKING_URI."""
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    base = root or MLRUNS
    base.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(base / 'mlflow.db').as_posix()}"


@dataclass(frozen=True)
class SimConfig:
    """SYNTHETIC freMTPL2-like book: monthly cohorts of policies with Tweedie claim cost."""

    n_months: int = 24
    policies_per_month: int = 25_000
    shift_start_month: int = 19  # injected covariate shift (young drivers, denser areas) from this month
    inflation_per_month: float = 0.005  # claims inflation on severity, compounding monthly
    seed: int = SEED


@dataclass(frozen=True)
class GatePolicy:
    """Champion/challenger promotion gate. All must hold for automatic promotion."""

    min_rel_deviance_gain: float = 0.0  # challenger deviance must be strictly lower than the champion's (gain > 0)
    min_gini: float = 0.20  # absolute floor on exposure-weighted normalised Gini
    ae_band: tuple[float, float] = (0.90, 1.10)  # actual / expected must sit in this band


@dataclass(frozen=True)
class TriggerPolicy:
    """Retraining trigger evaluated on each monthly production batch."""

    drift_share: float = 0.20  # share of monitored features flagged by Evidently
    ae_tolerance: float = 0.10  # |A/E - 1| beyond this triggers
    deviance_ratio: float = 1.10  # monthly deviance / reference deviance
    cooldown_months: int = 1  # months after a promotion during which triggers are suppressed
    train_window_months: int = 12


@dataclass(frozen=True)
class RunConfig:
    sim: SimConfig = field(default_factory=SimConfig)
    gate: GatePolicy = field(default_factory=GatePolicy)
    trigger: TriggerPolicy = field(default_factory=TriggerPolicy)
    initial_train_months: int = 12
    html_reports: tuple[int, ...] = (18, 19)  # months whose Evidently HTML is copied to results/reports
