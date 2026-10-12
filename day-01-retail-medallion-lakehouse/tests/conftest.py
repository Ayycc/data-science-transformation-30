"""Shared fixtures: build the lakehouse once from the committed SYNTHETIC sample."""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_lakehouse.config import SAMPLE_DIR  # noqa: E402
from retail_lakehouse.pipeline import run_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict:
    work = tmp_path_factory.mktemp("lakehouse")
    db = work / "sample.duckdb"
    summary = run_pipeline(SAMPLE_DIR, db, work / "target", docs=False)
    return {"db": db, "summary": summary, "landing": SAMPLE_DIR}


@pytest.fixture()
def con(built: dict):
    c = duckdb.connect(str(built["db"]), read_only=True)
    yield c
    c.close()
