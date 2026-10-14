"""Paths, scales and study parameters shared by the whole pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
SAMPLE_DIR = DATA_DIR / "sample"
RESULTS_DIR = ROOT / "results"
SQL_DIR = Path(__file__).resolve().parent / "sql"

SEED = 42

# Simulated calendar. Encounters are generated for the whole window.
WINDOW_START = date(2021, 1, 1)
WINDOW_END = date(2024, 12, 31)

# Cohort parameters (passed into the SQL).
LOOKBACK_DAYS = 365  # index discharges need one year of history
FOLLOWUP_DAYS = 30  # readmission horizon
TRAIN_END = date(2023, 12, 31)  # index discharges up to this date train the models; later ones are test

EHR_SOURCES = ("ehr_a", "ehr_b", "ehr_c")


@dataclass(frozen=True)
class Scale:
    name: str
    n_patients: int
    invalid_rate: float  # share of bundles deliberately corrupted to exercise validation
    seed: int


SCALES = {
    "full": Scale("full", n_patients=20_000, invalid_rate=0.01, seed=SEED),
    "sample": Scale("sample", n_patients=800, invalid_rate=0.03, seed=SEED + 1),
}
