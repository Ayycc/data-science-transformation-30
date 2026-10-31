"""make run: regenerate data, then run the full experiment and write results/."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regrag.data import build  # noqa: E402
from regrag.pipeline import run  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    prov = build(ROOT / "data" / "raw", ROOT / "data" / "sample")
    print(f"data path used: {prov['path_used']} ({prov['n_clauses']} clauses, {prov['n_questions']} questions)")
    m = run(ROOT / "data" / "raw", ROOT / "results")
    for c, a in m["acceptance_hybrid_vs_single"].items():
        print(
            f"[{c}] recall@5 hybrid={a['hybrid_recall@5']} bm25={a['bm25_recall@5']} dense={a['dense_recall@5']} beats_both={a['hybrid_beats_both']}"
        )
    print(json.dumps(m["generation"]["extractive_structure"], indent=2))
    print(f"done in {time.time() - t0:.1f}s")
