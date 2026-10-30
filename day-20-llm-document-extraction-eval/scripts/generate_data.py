"""make data: write the seeded SYNTHETIC corpus to data/raw/ (git-ignored) and data/sample/ (committed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract_eval.pipeline import ROOT, build_corpus  # noqa: E402

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    print(build_corpus(n, ROOT / "data" / "raw", ROOT / "data" / "sample"))
