"""Reproduce every reported result: generate data, validate, normalise, cohort, model, evaluate.

Usage: python scripts/run_pipeline.py --scale full   (what `make run` does)
       python scripts/run_pipeline.py --scale sample --results data/sample_results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_data import main as generate_main  # noqa: E402
from readmit.config import GENERATED_DIR, RESULTS_DIR, SAMPLE_DIR, SCALES  # noqa: E402
from readmit.pipeline import run, save_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=sorted(SCALES), default="full")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--reuse-data", action="store_true", help="skip regeneration if data/generated exists")
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    t0 = time.perf_counter()
    data_dir = SAMPLE_DIR if args.scale == "sample" else GENERATED_DIR
    if args.scale == "full" and not (args.reuse_data and (data_dir / "manifest.json").exists()):
        generate_main("full")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    truth_path = data_dir / "truth.csv.gz"
    truth = pd.read_csv(truth_path)[["key", "p_true"]] if truth_path.exists() else None
    metrics = run(data_dir, args.results, manifest, truth=truth, n_boot=args.n_boot)
    metrics["runtime_seconds"]["generate_s"] = manifest.get("generate_seconds")
    metrics["runtime_seconds"]["make_run_total"] = round(time.perf_counter() - t0, 1)
    save_metrics(metrics, args.results)
    print(f"[done] {metrics['runtime_seconds']['make_run_total']}s -> {args.results}", flush=True)


if __name__ == "__main__":
    main()
