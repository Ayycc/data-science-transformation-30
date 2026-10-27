"""Fetch NYC TLC trips if reachable, else write the seeded SYNTHETIC city to data/raw/ (+ data/sample/)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ridehail_fs.config import Paths, SimConfig  # noqa: E402
from ridehail_fs.data import acquire  # noqa: E402

if __name__ == "__main__":
    prov = acquire(SimConfig(), Paths(), allow_network="--offline" not in sys.argv)
    print(json.dumps({k: prov[k] for k in ("source", "tlc_status", "rows")}, indent=2))
