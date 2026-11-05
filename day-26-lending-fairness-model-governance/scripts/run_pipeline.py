"""Entry point: ``python scripts/run_pipeline.py [--small] [--data auto|hmda|german|synthetic]``."""

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

from credit26.pipeline import RunConfig, run, small_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="tiny smoke run into data/smoke_results")
    ap.add_argument("--data", default=os.environ.get("CREDIT_DATA"), help="auto (default) | hmda | german | synthetic; env CREDIT_DATA")
    args = ap.parse_args()
    cfg = small_config() if args.small else RunConfig()
    if args.data:
        cfg.data_mode = args.data
    out = ROOT / ("data/smoke_results" if args.small else "results")
    t0 = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)

    res = run(cfg, out, ROOT / "data", sample_dir=None if args.small else ROOT / "data" / "sample", log=log)
    log(f"recommended model: {res['recommended']['model']}")
    log(f"done in {res['timings']['total_seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
