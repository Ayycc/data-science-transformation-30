"""Full lifecycle: data -> MLflow-tracked GLM/GBM -> gated registry promotion -> FastAPI scoring of monthly
batches -> Evidently drift/performance monitoring -> retraining trigger -> gated re-promotion -> results/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claims_mlops import pipeline  # noqa: E402

if __name__ == "__main__":
    m = pipeline.run(force_synthetic="--synthetic" in sys.argv)
    print(
        json.dumps(
            {k: m[k] for k in ("data", "initial_training", "retraining", "final_champion_version", "runtime_seconds")},
            indent=2,
            default=float,
        )
    )
