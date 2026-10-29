"""Write the SYNTHETIC raw domain feeds.

* ``data/raw/<domain>/*.csv``   full scale (90 days, about 420 orders/day; git-ignored)
* ``data/sample/<domain>/*.csv`` same 90-day timeline and schema evolution at ~6 orders/day (committed, used by tests)

There is no public source for this prompt: the brief asks for synthetic multi-domain data, so the
pipeline always uses the seeded generator (``path_used = synthetic_generator``) and makes no network calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mesh.config import RAW, SAMPLE, SEED  # noqa: E402
from mesh.generators import write_raw  # noqa: E402

SAMPLE_RATE = 6.0


def main() -> dict:
    full = write_raw(RAW)
    sample = write_raw(SAMPLE, base_rate=SAMPLE_RATE)
    size = sum(f.stat().st_size for f in SAMPLE.rglob("*.csv"))
    info = {"path_used": "synthetic_generator", "seed": SEED, "raw_rows": full, "sample_rows": sample, "sample_bytes": size}
    print(json.dumps(info))
    return info


if __name__ == "__main__":
    main()
