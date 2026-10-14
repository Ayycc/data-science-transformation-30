"""Generate SYNTHETIC FHIR-like bundles.

Usage: python scripts/generate_data.py --scale full     (20,000 patients -> data/generated, git-ignored)
       python scripts/generate_data.py --scale sample   (committed sample -> data/sample)

Before generating at full scale, the script probes the public Synthea sample-data download. The
probe result is recorded in the manifest (``data_path_used``). This build has no Synthea FHIR
adapter, so the synthetic generator is used either way; see the README.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from readmit.config import GENERATED_DIR, SAMPLE_DIR, SCALES  # noqa: E402
from readmit.generate import generate, write  # noqa: E402

SYNTHEA_URL = (
    "https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_fhir_latest.zip"
)


def probe_synthea(timeout: float = 10.0) -> str:
    try:
        req = urllib.request.Request(SYNTHEA_URL, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed public URL
            return f"reachable (HTTP {r.status})"
    except Exception as exc:  # noqa: BLE001 - any failure means fall back
        return f"unreachable ({type(exc).__name__}: {str(exc)[:120]})"


def main(scale_name: str, probe: bool = True) -> dict:
    scale = SCALES[scale_name]
    out = SAMPLE_DIR if scale_name == "sample" else GENERATED_DIR
    t0 = time.perf_counter()
    synthea = probe_synthea() if probe else "not probed"
    by_ehr, manifest, truth = generate(scale)
    manifest["synthea_probe"] = synthea
    manifest["data_path_used"] = "synthetic_generator"
    write(by_ehr, manifest, out, level=9 if scale_name == "sample" else 6)
    pd.DataFrame(truth).to_csv(out / "truth.csv.gz", index=False, float_format="%.5f")
    manifest["generate_seconds"] = round(time.perf_counter() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[data] {scale.n_patients:,} SYNTHETIC patients -> {out} in {manifest['generate_seconds']}s ({synthea})")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=sorted(SCALES), default="full")
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()
    main(a.scale, probe=not a.no_probe and a.scale == "full")
