"""Prompt/version registry. Every prompt file is pinned by SHA-256 in registry.json; editing a prompt without
registering a new version fails ``load_prompt`` (and the regression test), so results always name the exact prompt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent


def registry() -> dict[str, dict[str, str]]:
    return json.loads((HERE / "registry.json").read_text(encoding="utf-8"))


def load_prompt(version: str) -> str:
    reg = registry()
    if version not in reg:
        raise KeyError(f"prompt version {version!r} is not registered")
    path = HERE / reg[version]["file"]
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != reg[version]["sha256"]:
        raise ValueError(
            f"prompt {version} changed on disk (sha256 {digest[:12]} != registered {reg[version]['sha256'][:12]}); register a new version"
        )
    return data.decode("utf-8")
