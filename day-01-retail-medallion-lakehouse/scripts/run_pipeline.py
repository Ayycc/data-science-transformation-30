"""Reproduce every reported result: generate data, build the lakehouse, analyse.

Usage: python scripts/run_pipeline.py --scale full   (what `make run` does)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_lakehouse.analysis import analyse  # noqa: E402
from retail_lakehouse.config import (  # noqa: E402
    DEFAULT_DB_PATH,
    GENERATED_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    SAMPLE_DIR,
    SCALES,
    SEED,
)
from retail_lakehouse.generate import generate  # noqa: E402
from retail_lakehouse.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=sorted(SCALES), default="full")
    ap.add_argument(
        "--landing",
        type=Path,
        default=None,
        help="existing landing dir to use instead of generating (e.g. data/sample)",
    )
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--target", type=Path, default=PROJECT_ROOT / "dbt" / "target")
    ap.add_argument("--analysis-only", action="store_true", help="re-run only the analysis against an existing build")
    args = ap.parse_args()
    summary_path = args.target / "pipeline_summary.json"

    t0 = time.perf_counter()
    gen_s = 0.0
    if args.landing is None and not args.analysis_only:
        landing = SAMPLE_DIR if args.scale == "sample" else GENERATED_DIR
        if args.scale != "sample":
            s = time.perf_counter()
            generate(landing, SCALES[args.scale], seed=SEED)
            gen_s = time.perf_counter() - s
            print(f"[1/3] generated {args.scale} landing zone in {gen_s:.1f}s", flush=True)
    else:
        landing = args.landing or (SAMPLE_DIR if args.scale == "sample" else GENERATED_DIR)
    if args.analysis_only:
        summary = json.loads(summary_path.read_text())
    else:
        summary = run_pipeline(landing, args.db, args.target)
        print(f"[2/3] pipeline finished: {json.dumps(summary['timings'])}", flush=True)
        summary["timings"]["generate_s"] = round(gen_s, 2)
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
    metrics = analyse(args.db, landing, args.results, summary)
    metrics["timings_s"]["end_to_end_s"] = round(time.perf_counter() - t0, 2)
    (args.results / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    print(f"[3/3] results written to {args.results} in {metrics['timings_s']['end_to_end_s']:.1f}s total")


if __name__ == "__main__":
    main()
