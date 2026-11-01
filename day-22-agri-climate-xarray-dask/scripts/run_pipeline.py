"""Entry point for `make run`: regenerates the data and every number in results/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agriclimate.config import RunConfig  # noqa: E402
from agriclimate.pipeline import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="tiny configuration (smoke test)")
    ap.add_argument("--offline", action="store_true", help="skip the public-data probe")
    ap.add_argument("--no-out-of-core", action="store_true")
    ap.add_argument("--keep-big", action="store_true", help="keep the multi-GB out-of-core cube on disk")
    a = ap.parse_args()
    cfg = RunConfig.small() if a.small else RunConfig()
    run(cfg, ROOT, try_public=not a.offline, out_of_core=not a.no_out_of_core, keep_big=a.keep_big)
