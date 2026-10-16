"""Run the full pipeline. `make run` uses --scale full and writes every reported number to results/."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from civic311.config import RESULTS_DIR, SCALES  # noqa: E402
from civic311.pipeline import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=sorted(SCALES), default="full")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    a = ap.parse_args()
    m = run(SCALES[a.scale], a.results.resolve(), probe=a.scale == "full")
    q, d = m["quarantine"], m["defect_detection"]
    print(f"rows {m['data']['rows_raw']:,} | quarantined {q['rows_quarantined']:,} ({q['pct_quarantined']}%)")
    print(f"defect classes fully caught {d['classes_fully_caught']}/{d['classes_present']}; "
          f"clean rows wrongly quarantined {d['clean_rows_quarantined']}")  # fmt: skip
    print(f"hotspots {m['hotspots']['gi_classes']} | total {m['timings_s']['total']} s")
