"""Entry point: ``python scripts/run_pipeline.py [--small] [--data auto|movielens|synthetic]``."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit's ALS asks for single-threaded BLAS
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys24.pipeline import RunConfig, run, small_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="tiny smoke run into data/smoke_results")
    ap.add_argument("--data", default=None, help="auto (default) | movielens | synthetic; env RECSYS_DATA")
    args = ap.parse_args()
    cfg = small_config() if args.small else RunConfig()
    if args.data:
        cfg.data_mode = args.data
    out = ROOT / ("data/smoke_results" if args.small else "results")
    t0 = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)

    res = run(cfg, out, ROOT / "data", sample_dir=None if args.small else ROOT / "data" / "sample", log=log)
    t = res["offline"]["test"]
    log(f"NDCG@10 ALS {t['ALS']['ndcg@10']:.4f} vs two-tower+re-ranker {t['Two-tower + re-ranker']['ndcg@10']:.4f}")
    log(f"done in {res['timings']['total_seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
