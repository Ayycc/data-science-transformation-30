"""Data acquisition: probe the public EIA-930 source, fall back to the SYNTHETIC generator."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

from .config import EIA930_PROBE_URL, GENERATED_DIR, SAMPLE_DIR, Scale
from .generate import generate


def probe_public_source(url: str = EIA930_PROBE_URL, timeout: float = 15.0) -> dict:
    """Try to reach the public EIA-930 bulk file (no API key). Never raises."""
    try:
        req = urllib.request.Request(url, method="GET", headers={"Range": "bytes=0-2047"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            head = r.read(2048).decode("utf-8", errors="replace")
        return {"url": url, "reachable": True, "detail": f"HTTP {r.status}, first line: {head.splitlines()[0][:120]}"}
    except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
        return {"url": url, "reachable": False, "detail": f"{type(exc).__name__}: {exc}"}


def data_path(scale: Scale) -> Path:
    return (SAMPLE_DIR if scale.name == "sample" else GENERATED_DIR) / "load.csv.gz"


def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    """Deterministic gzip (mtime=0) so the committed sample is byte-reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M", compression={"method": "gzip", "mtime": 0})


def read_load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ds"])
    df["is_holiday"] = df["is_holiday"].astype("int8")
    return df


def acquire(scale: Scale, probe: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return the load frame and a manifest describing which path produced it.

    The project has no EIA-930 adapter (see README, Limitations): the probe only records whether the
    public source was reachable from the build machine. Data always comes from the seeded generator.
    """
    manifest: dict = {"scale": scale.name, "seed": scale.seed}
    manifest["public_probe"] = probe_public_source() if probe else {"skipped": True}
    df = generate(scale)
    manifest.update(
        {
            "path_used": "synthetic_generator",
            "start": scale.start,
            "end_exclusive": scale.end,
            "regions": list(scale.regions),
            "rows": int(len(df)),
            "synthetic": True,
        }
    )
    out = data_path(scale)
    write_csv_gz(df, out)
    manifest["file"] = str(out.relative_to(out.parents[2]))
    manifest["file_bytes"] = out.stat().st_size
    (out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return df, manifest
