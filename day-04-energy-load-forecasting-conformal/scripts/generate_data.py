"""Generate the SYNTHETIC load data. `--scale sample` rewrites the committed sample in data/sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridcast.config import SCALES  # noqa: E402
from gridcast.data import acquire  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=sorted(SCALES), default="full")
    a = ap.parse_args()
    scale = SCALES[a.scale]
    df, manifest = acquire(scale, probe=scale.name == "full")
    print(
        f"{manifest['path_used']}: {manifest['rows']:,} rows -> {manifest['file']} ({manifest['file_bytes']:,} bytes)"
    )
    print(f"public probe: {manifest['public_probe']}")
