"""Generate the SYNTHETIC landing zone (data/generated/landing) and the committed sample (data/sample)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clickstream.config import PRESETS, TEST
from clickstream.data import write_landing, write_sample
from clickstream.spark import get_spark

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full", choices=sorted(PRESETS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sc = PRESETS[a.scale]
    spark = get_spark(shuffle_partitions=sc.shuffle_partitions, ui=False)
    n = write_sample(spark, TEST, ROOT / "data" / "sample" / "deliveries_sample.csv.gz")
    print(f"sample: {n} SYNTHETIC deliveries -> data/sample/deliveries_sample.csv.gz")
    info = write_landing(spark, sc, ROOT / "data" / "generated" / "landing", force=a.force)
    print(json.dumps(info, indent=2))
