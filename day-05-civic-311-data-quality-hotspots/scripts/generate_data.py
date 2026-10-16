"""Generate the SYNTHETIC 311 extract. `--scale sample` rewrites the committed sample in data/sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from civic311.config import SCALES  # noqa: E402
from civic311.generate import write  # noqa: E402
from civic311.pipeline import probe_public  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=sorted(SCALES), default="full")
    a = ap.parse_args()
    scale = SCALES[a.scale]
    m = write(scale, public_probe=probe_public() if scale.name == "full" else None)
    print(f"{m['path_used']}: {m['rows']:,} rows -> {m['files']['raw']} ({m['raw_bytes']:,} bytes)")
