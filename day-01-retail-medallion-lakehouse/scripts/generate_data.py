"""Generate the SYNTHETIC landing zone. Usage: python scripts/generate_data.py --scale full."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_lakehouse.config import GENERATED_DIR, SCALES, SEED  # noqa: E402
from retail_lakehouse.generate import generate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=sorted(SCALES), default="full")
    ap.add_argument("--out", type=Path, default=GENERATED_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    t0 = time.perf_counter()
    manifest = generate(args.out, SCALES[args.scale], seed=args.seed)
    print(json.dumps(manifest, indent=2))
    print(f"generated {args.scale} landing zone in {time.perf_counter() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
