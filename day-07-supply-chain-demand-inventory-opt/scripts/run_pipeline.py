"""Reproduce every number and figure in README.md / results/ (SYNTHETIC data)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from invopt.config import load_config
from invopt.pipeline import run

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--skus", type=int, default=None, help="override the number of SKUs (default: config)")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--data", default=str(ROOT / "data" / "generated"))
    ap.add_argument("--no-sensitivity", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    sample = ROOT / "data" / "sample" if a.skus is None else None
    m = run(cfg, Path(a.results), Path(a.data), sample_dir=sample, n_skus=a.skus, sensitivity=not a.no_sensitivity)
    print(json.dumps(m["headline"], indent=2))
    print(json.dumps(m["simulation"], indent=2))
    print(json.dumps(m["milp"], indent=2))
    print(m["runtime_seconds"])
