"""Snapshot GitHub Actions results per day folder into data/sample/ci_snapshot.json (needs `gh`).

This is the only network step for the portfolio report and it is NOT part of `make run`: the report
reads the committed snapshot so it builds offline. Refresh with `make ci-snapshot`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = "Ayycc/data-science-transformation-30"
OUT = Path(__file__).resolve().parents[1] / "data" / "sample" / "ci_snapshot.json"


def gh(path: str) -> dict:
    out = subprocess.run(["gh", "api", path], check=True, capture_output=True, text=True, timeout=60)
    return json.loads(out.stdout)


def main() -> None:
    runs = gh(f"repos/{REPO}/actions/runs?per_page=100")["workflow_runs"]
    per_day: dict[str, dict] = {}
    for r in sorted(runs, key=lambda r: r["created_at"]):
        jobs = gh(f"repos/{REPO}/actions/runs/{r['id']}/jobs?per_page=100")["jobs"]
        for j in jobs:
            m = re.search(r"(day-\d{2}-[^\s,)]+)", j["name"])
            if not m:
                continue
            per_day[m.group(1)] = {  # later runs overwrite earlier ones
                "conclusion": j["conclusion"],
                "run_id": r["id"],
                "head_sha": r["head_sha"][:7],
                "run_title": r["display_title"][:90],
                "completed_at": j["completed_at"],
            }
    snap = {
        "fetched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": f"GitHub Actions API for {REPO} via gh CLI (workflow CI, job 'test (<day folder>)')",
        "runs_scanned": len(runs),
        "days": dict(sorted(per_day.items())),
    }
    OUT.write_text(json.dumps(snap, indent=2) + "\n")
    print(f"wrote {OUT} with {len(per_day)} day folders from {len(runs)} runs", file=sys.stderr)


if __name__ == "__main__":
    main()
