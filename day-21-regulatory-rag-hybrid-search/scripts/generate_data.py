"""Build the corpus + gold set: public eCFR text if reachable, else the seeded SYNTHETIC generator."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regrag.data import build  # noqa: E402

if __name__ == "__main__":
    prov = build(ROOT / "data" / "raw", ROOT / "data" / "sample")
    print(json.dumps(prov, indent=2))
