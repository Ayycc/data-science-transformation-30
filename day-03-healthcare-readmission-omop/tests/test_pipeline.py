"""Generator determinism and an end-to-end smoke run on the committed sample."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from readmit.config import SAMPLE_DIR, SCALES
from readmit.generate import generate, write
from readmit.pipeline import run


def test_generator_reproduces_committed_sample(tmp_path: Path) -> None:
    by_ehr, manifest, _ = generate(SCALES["sample"])
    write(by_ehr, manifest, tmp_path, level=9)
    for ehr in by_ehr:
        new = gzip.decompress((tmp_path / f"{ehr}.ndjson.gz").read_bytes())
        old = gzip.decompress((SAMPLE_DIR / f"{ehr}.ndjson.gz").read_bytes())
        assert new == old, ehr
    assert manifest["corrupted_bundles"] == json.loads((SAMPLE_DIR / "manifest.json").read_text())["corrupted_bundles"]


def test_sample_is_small() -> None:
    assert sum(p.stat().st_size for p in SAMPLE_DIR.iterdir()) <= 1_000_000


def test_end_to_end_smoke(tmp_path: Path) -> None:
    manifest = json.loads((SAMPLE_DIR / "manifest.json").read_text())
    m = run(SAMPLE_DIR, tmp_path, manifest, n_boot=20)
    assert m["validation"]["injected_defects_rejected"] == m["validation"]["injected_defects"]
    assert m["primary_model"] in {"logistic_l2", "gbm_calibrated"}
    for k in ("logistic_l2", "gbm_calibrated", "lace"):
        assert 0.5 < m["test"][k]["auroc"] <= 1.0
        assert 0 < m["test"][k]["brier"] < 0.25
    for f in ("calibration.png", "decision_curve.png", "subgroup_performance.png", "report.md"):
        assert (tmp_path / f).exists()
