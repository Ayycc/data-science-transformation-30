"""Single reproducible command for the whole backtest: data -> models -> conformal -> results/."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridcast.backtest import run_backtest  # noqa: E402
from gridcast.config import RESULTS_DIR, SCALES  # noqa: E402
from gridcast.data import acquire, data_path, read_load  # noqa: E402
from gridcast.report import write_results  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=sorted(SCALES), default="full")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    a = ap.parse_args()
    scale = SCALES[a.scale]
    t0 = time.time()
    if scale.name == "full":
        df, manifest = acquire(scale, probe=True)  # regenerate every run; never committed
    else:
        import json

        df = read_load(data_path(scale))  # the committed sample
        manifest = json.loads((data_path(scale).parent / "manifest.json").read_text())
    t_data = time.time() - t0
    print(f"data: {manifest['path_used']}, {len(df):,} rows, probe={manifest['public_probe']}")
    res = run_backtest(df, scale, log=lambda m: print(m, flush=True), stress=scale.name == "full")
    runtime = {"data_s": t_data, **res["timings"]}
    runtime["total_s"] = time.time() - t0
    m = write_results(res, df, manifest, a.results, runtime)
    acc = m["acceptance"]
    print(f"selected conformal: {m['conformal_selection']['selected']}")
    print(
        f"coverage {acc['coverage_overall']:.3f} "
        f"(regions {acc['coverage_min_region']:.3f}-{acc['coverage_max_region']:.3f}); "
        f"best {acc['best_model']} MASE {acc['best_model_mase']:.3f} vs {acc['best_seasonal_naive']} "
        f"{acc['seasonal_naive_mase']:.3f} ({acc['mase_improvement_vs_seasonal_naive_pct']:.1f}% better)"
    )
    print(f"total {runtime['total_s']:.0f}s -> {a.results}")
