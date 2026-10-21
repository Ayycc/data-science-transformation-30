"""`make run`: full end-to-end run (public Bitext corpus if reachable, else SYNTHETIC)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ticket_intel.config import FULL  # noqa: E402
from ticket_intel.pipeline import run  # noqa: E402

if __name__ == "__main__":
    run(FULL)
