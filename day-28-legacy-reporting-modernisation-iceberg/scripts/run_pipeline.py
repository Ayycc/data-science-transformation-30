"""Entry point for `make run` / `make data`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legacy28.pipeline import main  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="smoke run: 2,000 policies -> data/smoke_results")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.small:
        main(n_policies=2_000, scales=[1, 2], reps=1, results=root / "data" / "smoke_results", data=root / "data" / "smoke")
    else:
        main()
