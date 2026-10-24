"""Write the committed SYNTHETIC sample (data/sample/) and the full SYNTHETIC stream to data/raw/ (git-ignored)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb  # noqa: E402

from iot_stream.config import SimConfig  # noqa: E402
from iot_stream.data import SAMPLE_CONFIG, probe_cmapss, save_sample  # noqa: E402
from iot_stream.simulate import simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    print("C-MAPSS probe:", probe_cmapss())
    sizes = save_sample(simulate(SAMPLE_CONFIG))
    print("sample written:", sizes)
    if "--sample-only" not in sys.argv:
        stream = simulate(SimConfig())
        raw = ROOT / "data" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        df = stream.to_frame()  # noqa: F841 (read by DuckDB replacement scan)
        duckdb.sql(f"COPY df TO '{raw / 'events.parquet'}' (FORMAT parquet, COMPRESSION zstd)")
        stream.failures.to_csv(raw / "failures.csv", index=False)
        print(f"full SYNTHETIC stream: {len(stream):,} events -> {raw}")
