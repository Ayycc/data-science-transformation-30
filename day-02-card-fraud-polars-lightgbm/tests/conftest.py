"""Shared fixtures built from the committed SYNTHETIC sample (data/sample)."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from card_fraud.config import SAMPLE_DIR  # noqa: E402
from card_fraud.features_polars import build_features  # noqa: E402


@pytest.fixture(scope="session")
def raw() -> pl.DataFrame:
    return pl.read_parquet(SAMPLE_DIR / "transactions.parquet")


@pytest.fixture(scope="session")
def feats(raw: pl.DataFrame) -> pl.DataFrame:
    return build_features(raw)
