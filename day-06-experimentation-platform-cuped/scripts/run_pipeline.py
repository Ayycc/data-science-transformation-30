"""Entry point for `make run`: simulate, analyse, write readouts and run the Monte Carlo study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expengine.pipeline import FULL, SMALL, run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=["full", "small"], default="full")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--data", default=str(ROOT / "data" / "generated"))
    a = ap.parse_args()
    run(FULL if a.scale == "full" else SMALL, Path(a.results), Path(a.data))


if __name__ == "__main__":
    main()
