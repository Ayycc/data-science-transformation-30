"""Write the SYNTHETIC month to data/ (git-ignored) and refresh the committed sample in data/sample/.

Usage: python scripts/generate_data.py [--preset full|tiny] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from air_quality.config import PRESETS, data_dir  # noqa: E402
from air_quality.generate import write_dataset  # noqa: E402
from air_quality.sample import write_sample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full", choices=sorted(PRESETS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else data_dir()
    summary = write_dataset(PRESETS[args.preset], out)
    if args.preset == "full" and args.out is None:
        files = write_sample(ROOT / "data" / "sample")
        summary["sample_files"] = files
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
