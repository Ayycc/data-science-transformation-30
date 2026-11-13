"""Follow-up check: does refitting the two-tower on train+val (as ALS is) close the NDCG@10 gap to ALS?

The main pipeline keeps the two-tower trained on train only (production-realistic, keeps the re-ranker's tt_score
on the scale it was trained on). This script reruns the offline suite with the two-tower also refit for test, so the
comparison with ALS is like for like. Same world, seed, config and ALS selection; the test split is only scored.
Writes results/refit_check.json.  Usage: ``python scripts/refit_check.py``  (~5 min on 4 CPU cores).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys24.data import load_dataset  # noqa: E402
from recsys24.pipeline import RunConfig, offline_suite  # noqa: E402

NAMES = ("ALS", "Two-tower (exact)", "Two-tower + re-ranker", "Two-tower + re-ranker (+ALS feature)")


def main() -> None:
    cfg = RunConfig()
    cfg.data_mode = "synthetic"
    t0 = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)

    ds, _ = load_dataset(cfg.data_mode, ROOT / "data" / "raw", cfg.world)
    out = {}
    for refit in (False, True):
        res, _ = offline_suite(ds, cfg, log, refit_two_tower=refit)
        key = "two_tower_refit_on_train_val" if refit else "two_tower_train_only"
        out[key] = {
            "ndcg@10": {n: res["test"][n]["ndcg@10"] for n in NAMES},
            "recall@10": {n: res["test"][n]["recall@10"] for n in NAMES},
            "rerank_minus_als": res["ndcg10_rerank_minus_als"],
        }
        log(f"{key}: " + ", ".join(f"{n} {v:.4f}" for n, v in out[key]["ndcg@10"].items()))
    (ROOT / "results" / "refit_check.json").write_text(json.dumps(out, indent=2))
    log("wrote results/refit_check.json")


if __name__ == "__main__":
    main()
