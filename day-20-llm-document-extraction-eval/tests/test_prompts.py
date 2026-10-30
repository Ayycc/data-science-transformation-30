import hashlib
import json

import pytest

import extract_eval.prompts as prompts


def test_every_registered_prompt_matches_its_hash():
    reg = prompts.registry()
    assert "extract_v1" in reg
    for version, entry in reg.items():
        text = prompts.load_prompt(version)
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == entry["sha256"]
        assert "null" in text and "field_confidence" in text


def test_prompt_files_are_all_registered():
    files = {p.name for p in prompts.HERE.glob("*.txt")}
    assert files == {e["file"] for e in prompts.registry().values()}


def test_edited_prompt_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "p.txt").write_text("v1")
    (tmp_path / "registry.json").write_text(json.dumps({"p": {"file": "p.txt", "sha256": hashlib.sha256(b"v1").hexdigest()}}))
    monkeypatch.setattr(prompts, "HERE", tmp_path)
    assert prompts.load_prompt("p") == "v1"
    (tmp_path / "p.txt").write_text("v1 edited")
    with pytest.raises(ValueError, match="register a new version"):
        prompts.load_prompt("p")
    with pytest.raises(KeyError):
        prompts.load_prompt("missing")
