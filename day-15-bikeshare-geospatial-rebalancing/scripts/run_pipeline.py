"""Run the full Day 15 pipeline and write results/ (``make run``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bikeshare import pipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="skip the public download and use the SYNTHETIC generator")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default results/)")
    args = ap.parse_args()
    pipeline.run(force_synthetic=args.synthetic, **({"out_dir": args.out} if args.out else {}))
