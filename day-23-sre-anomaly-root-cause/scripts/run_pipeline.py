"""Entry point for `make run` / `make data`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sreanomaly.generate import SimConfig  # noqa: E402
from sreanomaly.pipeline import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="smoke run: 5 days, 12 incidents, no public check, writes to data/smoke_results")
    ap.add_argument("--no-public", action="store_true", help="skip the optional NAB public-data sanity check")
    a = ap.parse_args()
    if a.small:
        cfg = SimConfig(n_days=5, n_incidents=12, n_concurrent=2, n_surges=1)
        m = run(cfg, results_dir=ROOT / "data" / "smoke_results", data_dir=None, public=False)
    else:
        m = run(SimConfig(), results_dir=ROOT / "results", data_dir=ROOT / "data", public=not a.no_public)
    m.pop("_rows", None)
    rc = m["root_cause"]["methods"]
    print(json.dumps({"alert_volume": m["alert_volume"], "ensemble+deploy_prior": rc["ensemble+deploy_prior"], "runtime": m["runtime"]}, indent=1))


if __name__ == "__main__":
    main()
