"""Write the SYNTHETIC panel to data/raw/ (git-ignored) and data/sample/ (committed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayes_mmm.data import get_panel, save  # noqa: E402

if __name__ == "__main__":
    panel, path_used = get_panel()
    paths = save(panel)
    print(f"{path_used}: {panel.sales.shape[0]} weeks x {panel.sales.shape[1]} regions (SYNTHETIC)")
    for k, p in paths.items():
        print(f"  {k}: {p} ({p.stat().st_size / 1024:.0f} KB)")
