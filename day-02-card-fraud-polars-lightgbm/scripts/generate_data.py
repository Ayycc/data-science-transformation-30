"""Generate the SYNTHETIC transaction stream.

Usage: python scripts/generate_data.py --scale full|medium|sample [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from card_fraud.config import GENERATED_DIR, SAMPLE_DIR, SCALES, SEED  # noqa: E402
from card_fraud.generate import generate, write  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=sorted(SCALES), default="full")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (SAMPLE_DIR if args.scale == "sample" else GENERATED_DIR)
    t = time.perf_counter()
    df, manifest = generate(SCALES[args.scale], seed=SEED)
    path = write(df, manifest, out)
    print(
        f"wrote {df.height:,} SYNTHETIC transactions ({manifest['fraud_rate']:.3%} fraud) to {path} "
        f"in {time.perf_counter() - t:.1f}s"
    )


if __name__ == "__main__":
    main()
