"""Generate the SYNTHETIC subscriber panel (data/generated, git-ignored) and the committed sample."""

from __future__ import annotations

import argparse
from pathlib import Path

from retention.pipeline import load_config, make_data, write_sample

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "generated"))
    a = ap.parse_args()
    panel, truth = make_data(load_config(a.config))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "panel.csv.gz", index=False)
    truth.to_csv(out / "campaign_truth.csv.gz", index=False)
    write_sample(panel, truth, ROOT / "data" / "sample")
    print(f"SYNTHETIC panel: {len(panel):,} customers, {int(panel['in_campaign'].sum()):,} in the campaign -> {out}")
