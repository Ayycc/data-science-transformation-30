"""Load the public Bitext corpus (download + cache) or fall back to the SYNTHETIC generator."""

from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path

import pandas as pd

from .augment import augment_frame
from .config import BITEXT_FILE, BITEXT_HF_URL, BITEXT_URL, RAW_DIR, SAMPLE_DIR, RunConfig
from .synth import generate_base

SOURCE_BITEXT = "PUBLIC: Bitext customer-support corpus (CDLA-Sharing-1.0) + SYNTHETIC PII/snippet layer"
SOURCE_SYNTH = "SYNTHETIC: seeded 15-intent SaaS ticket generator"
SOURCE_SAMPLE = "SAMPLE: committed data/sample/tickets_sample.csv"


def download_bitext(raw_dir: Path = RAW_DIR, timeout: float = 60.0) -> tuple[Path | None, str]:
    """Return (path, note). Uses the cache, then Hugging Face, then the pinned GitHub commit."""
    path = raw_dir / BITEXT_FILE
    if path.exists() and path.stat().st_size > 1_000_000:
        return path, "cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    notes = []
    for label, url in (("huggingface", BITEXT_HF_URL), ("github@pinned-commit", BITEXT_URL)):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - fixed https URLs
                payload = r.read()
            pd.read_csv(io.BytesIO(payload), nrows=5)  # sanity check
            path.write_bytes(payload)
            return path, label
        except Exception as e:  # noqa: BLE001 - any failure triggers the next source
            notes.append(f"{label}: {type(e).__name__}: {str(e)[:80]}")
    return None, " | ".join(notes)


def load_bitext_base(path: Path, max_per_intent: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"instruction": "text"})[["text", "intent", "category"]]
    df["category"] = df["category"].str.upper()
    if max_per_intent:
        df = df.groupby("intent", group_keys=False).sample(n=max_per_intent, random_state=seed, replace=False)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_tickets(cfg: RunConfig, source: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (tickets with gold PII spans, provenance dict).

    `source` (or env TICKET_SOURCE): auto | bitext | synthetic | sample.
    """
    source = (source or os.environ.get("TICKET_SOURCE", "auto")).lower()
    prov: dict = {"requested": source}
    if source == "sample":
        df = pd.read_csv(SAMPLE_DIR / "tickets_sample.csv")
        prov.update(path="sample", data_source=SOURCE_SAMPLE)
        return df, prov
    base = None
    if source in ("auto", "bitext"):
        path, note = download_bitext()
        prov["download"] = note
        if path is not None:
            base = load_bitext_base(path, cfg.max_per_intent, cfg.seed)
            prov.update(path="bitext", data_source=SOURCE_BITEXT)
        elif source == "bitext":
            raise RuntimeError(f"Bitext unavailable: {note}")
    if base is None:
        base = generate_base(cfg.synthetic_per_intent, cfg.seed)
        prov.update(path="synthetic", data_source=SOURCE_SYNTH)
    df = augment_frame(base, cfg.seed + 1, cfg.pii_frac, cfg.multilingual_frac)
    return df, prov
