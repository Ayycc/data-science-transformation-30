"""Shared fixtures: the committed SYNTHETIC sample (data/sample) run end to end into a temp results folder."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def sample_run(tmp_path_factory):
    from civic311.config import SAMPLE
    from civic311.pipeline import run

    out = tmp_path_factory.mktemp("sample") / "results"
    metrics = run(SAMPLE, out, probe=False, log=lambda m: None)
    return metrics, out
