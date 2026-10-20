"""Shared fixtures: a tiny SYNTHETIC month (20 stations x 30 days) generated once per session."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from air_quality.config import TINY  # noqa: E402
from air_quality.generate import write_dataset  # noqa: E402


@pytest.fixture(scope="session")
def tiny_source(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("tiny_source")
    write_dataset(TINY, out)
    return out


@pytest.fixture()
def tiny_data(tiny_source, tmp_path) -> Path:
    """A private copy of the landing zone (late arrivals not yet delivered) for one test."""
    dst = tmp_path / "data"
    for sub in ("landing", "late_arrivals", "reference"):
        shutil.copytree(tiny_source / sub, dst / sub)
    return dst


def deliver_late(data: Path) -> list[str]:
    parts = []
    for f in sorted((data / "late_arrivals").glob("*/batch_*.parquet")):
        dest = data / "landing" / f.parent.name
        shutil.copy2(f, dest / f.name)
        parts.append(f.parent.name)
    return parts
