"""Entry point: ``python scripts/run_pipeline.py [--small] [--workers N]``."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pricing27.pipeline import RunConfig, run, small_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="tiny smoke run into data/smoke_results")
    ap.add_argument(
        "--workers", type=int, default=int(os.environ.get("WORKERS", "4")), help="processes for the seed loop (results do not depend on it)"
    )
    args = ap.parse_args()
    cfg = small_config() if args.small else RunConfig()
    cfg.workers = args.workers
    out = ROOT / ("data/smoke_results" if args.small else "results")
    t0 = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)

    res = run(cfg, out, ROOT / "data" / "generated", sample_dir=None if args.small else ROOT / "data" / "sample", log=log)
    log(f"done in {res['timings']['total_seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
