"""make run entry point. ``--quick`` runs on the committed sample only (seconds)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_stream.pipeline import run  # noqa: E402

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    out = Path(__file__).resolve().parents[1] / ("results" if not quick else "data/quick_results")
    run(quick=quick, results_dir=out)
