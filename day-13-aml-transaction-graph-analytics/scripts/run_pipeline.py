"""Run the full Day 13 pipeline and write results/ (metrics.json, report.md, figures, alert HTML)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml_graph.pipeline import run  # noqa: E402

if __name__ == "__main__":
    m = run()
    fs = m["models"]["feature_sets"]
    print(f"PR-AUC tabular {fs['tabular']['pr_auc']:.3f} vs graph {fs['graph (all)']['pr_auc']:.3f}; acceptance: {m['acceptance']}")
