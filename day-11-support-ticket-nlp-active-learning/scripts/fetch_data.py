"""`make data`: download + cache the Bitext corpus (pinned commit), or report the SYNTHETIC fallback."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ticket_intel.data import download_bitext  # noqa: E402

if __name__ == "__main__":
    path, note = download_bitext()
    if path is None:
        print(f"Bitext unreachable ({note}); `make run` will use the SYNTHETIC generator.")
    else:
        print(f"Bitext corpus ready at {path} (source: {note})")
