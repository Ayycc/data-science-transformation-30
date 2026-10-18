"""Reproduce every number and figure in README.md / results/ (SYNTHETIC data)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retention.pipeline import load_config, run

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--customers", type=int, default=None, help="override the number of customers (default: config)")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--data", default=str(ROOT / "data" / "generated"))
    ap.add_argument("--fast", action="store_true", help="tiny forests (smoke test only)")
    a = ap.parse_args()
    m = run(load_config(a.config), Path(a.results), Path(a.data), n_customers=a.customers, fast=a.fast)
    print(json.dumps(m["headline"], indent=2))
    print(json.dumps(m["runtime_seconds"], indent=2))
