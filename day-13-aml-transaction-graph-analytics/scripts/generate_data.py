"""Write the full SYNTHETIC graph to data/raw/ (git-ignored) and refresh the small committed sample.

Usage: python scripts/generate_data.py [--sample-only]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml_graph.config import RAW_DIR, SAMPLE_DIR, SMALL, SimConfig  # noqa: E402
from aml_graph.data import probe_ibm  # noqa: E402
from aml_graph.simulate import simulate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    small = simulate(SMALL)
    small.accounts.to_csv(SAMPLE_DIR / "accounts.csv.gz", index=False)
    small.transactions.round({"ts": 0}).to_csv(SAMPLE_DIR / "transactions.csv.gz", index=False)
    small.instances.to_csv(SAMPLE_DIR / "instances.csv", index=False)
    print(f"sample: {len(small.accounts):,} accounts, {len(small.transactions):,} SYNTHETIC transactions -> {SAMPLE_DIR}")
    if args.sample_only:
        return
    ok, note = probe_ibm()
    print(f"IBM AML public dataset probe: {note} -> {'reachable' if ok else 'unreachable, using SYNTHETIC generator'}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    full = simulate(SimConfig())
    full.accounts.to_parquet(RAW_DIR / "accounts.parquet", index=False)
    full.transactions.to_parquet(RAW_DIR / "transactions.parquet", index=False)
    full.instances.to_parquet(RAW_DIR / "instances.parquet", index=False)
    print(f"full: {len(full.accounts):,} accounts, {len(full.transactions):,} SYNTHETIC transactions -> {RAW_DIR}")


if __name__ == "__main__":
    main()
