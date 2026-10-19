"""Reproduce every number and figure in README.md / results/ (SYNTHETIC data)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clickstream.config import PRESETS
from clickstream.figures import make_all
from clickstream.pipeline import run
from clickstream.spark import get_spark

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full", choices=sorted(PRESETS))
    ap.add_argument("--data", default=str(ROOT / "data" / "generated"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--driver-memory", default="10g")
    a = ap.parse_args()
    sc = PRESETS[a.scale]
    spark = get_spark(shuffle_partitions=sc.shuffle_partitions, driver_memory=a.driver_memory)
    m = run(
        spark, sc, Path(a.data), Path(a.results), skew_repeats=2 if sc.name == "full" else 1, layout_repeats=3 if sc.name == "full" else 1
    )
    print("figures:", make_all(Path(a.results)))
    print(json.dumps(m["timings"], indent=2))
