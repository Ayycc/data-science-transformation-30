"""Shared paths and generator settings for the Day 01 lakehouse."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DBT_DIR = PROJECT_ROOT / "dbt"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
GENERATED_DIR = DATA_DIR / "generated"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DB_PATH = DATA_DIR / "warehouse" / "retail.duckdb"
CLOCK_CHANGES_SEED = DBT_DIR / "seeds" / "pos_clock_changes.csv"

SEED = 42
# Two full years of trading, half-open interval [START, END) in UTC.
START = pd.Timestamp("2024-10-01")
END = pd.Timestamp("2026-10-01")

WEB_STORE_ID = 0
TRAINING_STORE_ID = 999

# Phase 1 of `make run` loads ingestion batches up to and including this month into a
# fresh DuckDB file; phase 2 lands the remaining batches and re-runs dbt incrementally.
PHASE1_LAST_BATCH = "2026-08"


@dataclass(frozen=True)
class Scale:
    """Size of a generated dataset."""

    name: str
    n_lines: int
    n_customers: int
    n_products: int
    n_stores: int = 120


FULL = Scale(name="full", n_lines=2_000_000, n_customers=50_000, n_products=5_000)
SAMPLE = Scale(name="sample", n_lines=8_000, n_customers=1_000, n_products=200)
SCALES = {s.name: s for s in (FULL, SAMPLE)}
