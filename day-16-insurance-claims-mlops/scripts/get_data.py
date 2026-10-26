"""Acquire data: public freMTPL2 via OpenML if reachable, else the seeded SYNTHETIC generator.

Writes data/raw/policies.parquet (git-ignored) and refreshes data/sample/policies_sample.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claims_mlops import data  # noqa: E402

if __name__ == "__main__":
    acq = data.load_or_create(force_synthetic="--synthetic" in sys.argv)
    data.write_sample(acq.policies)
    print(json.dumps({k: v for k, v in acq.provenance.items() if k != "sim"}, indent=2))
