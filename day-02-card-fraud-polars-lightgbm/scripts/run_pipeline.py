"""Reproduce every reported result: generate data, build features, tune, threshold, explain.

Usage: python scripts/run_pipeline.py --scale full   (what `make run` does)
       python scripts/run_pipeline.py --scale sample --results data/sample_results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from card_fraud.config import DATA_DIR, GENERATED_DIR, RESULTS_DIR, SAMPLE_DIR, SCALES, SEED  # noqa: E402
from card_fraud.generate import generate, write  # noqa: E402
from card_fraud.pipeline import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=sorted(SCALES), default="full")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--reuse-data", action="store_true", help="load existing generated parquet instead of regenerating")
    args = ap.parse_args()
    scale = SCALES[args.scale]
    data_dir = SAMPLE_DIR if args.scale == "sample" else GENERATED_DIR
    t0 = time.perf_counter()
    gen_s = 0.0
    if args.scale == "sample" or args.reuse_data:
        raw = pl.read_parquet(data_dir / "transactions.parquet")
        manifest = json.loads((data_dir / "manifest.json").read_text())
    else:
        df, manifest = generate(scale, seed=SEED)
        write(df, manifest, data_dir)
        gen_s = time.perf_counter() - t0
        raw = df
        print(f"[data] generated {raw.height:,} SYNTHETIC transactions in {gen_s:.1f}s", flush=True)
    model_dir = DATA_DIR / ("models" if args.results == RESULTS_DIR else "models_" + args.scale)
    metrics = run(raw, manifest, scale, args.results, model_dir=model_dir, gen_seconds=gen_s)
    total = time.perf_counter() - t0
    metrics["runtime_seconds"]["make_run_total"] = round(total, 1)
    (args.results / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    print(f"[done] {total:.1f}s -> {args.results}", flush=True)


if __name__ == "__main__":
    main()
