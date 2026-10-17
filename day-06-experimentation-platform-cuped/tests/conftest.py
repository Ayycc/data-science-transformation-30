"""Shared fixtures. All data is SYNTHETIC."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "sample" / "e01_sample.csv.gz")


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


@pytest.fixture(scope="session")
def small_run(tmp_path_factory):
    from expengine.pipeline import SMALL, run

    base = tmp_path_factory.mktemp("small")
    metrics = run(SMALL, base / "results", base / "data", log=lambda m: None)
    return metrics, base
