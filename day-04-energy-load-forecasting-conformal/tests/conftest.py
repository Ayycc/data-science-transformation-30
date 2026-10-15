"""Shared fixtures: the committed SYNTHETIC sample (data/sample) and a small backtest on it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridcast.backtest import run_backtest  # noqa: E402
from gridcast.config import SAMPLE  # noqa: E402
from gridcast.data import data_path, read_load  # noqa: E402


@pytest.fixture(scope="session")
def sample_df():
    return read_load(data_path(SAMPLE))


@pytest.fixture(scope="session")
def sample_backtest(sample_df):
    return run_backtest(sample_df, SAMPLE, log=lambda m: None, stress=False)
