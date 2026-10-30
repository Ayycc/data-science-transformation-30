"""Validation gate in front of the downstream store.

Only records that re-validate against the Pydantic schema here are written to ``accepted.jsonl``.
Anything else goes to ``quarantine.jsonl`` (with its validation errors) for a human to key or fix.
The gate re-validates itself and never trusts an extractor's own ``valid`` flag.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .extractors.base import ExtractionResult
from .schemas import validate_record


class InvalidRecordError(ValueError):
    pass


class DownstreamStore:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.accepted_path = self.out_dir / "accepted.jsonl"
        self.quarantine_path = self.out_dir / "quarantine.jsonl"
        for p in (self.accepted_path, self.quarantine_path):
            p.write_text("", encoding="utf-8")
        self.n_accepted = self.n_quarantined = 0

    def write_accepted(self, res: ExtractionResult, review_fields: list[str] | None = None) -> dict[str, Any]:
        model, errors = validate_record(res.doc_type, res.record_payload())
        if model is None:
            raise InvalidRecordError(f"{res.doc_id}: refusing to write an invalid record: {errors}")
        row = {
            "doc_id": res.doc_id,
            "extractor": res.extractor,
            "record": json.loads(model.model_dump_json()),
            "fields_needing_review": review_fields or [],
        }
        with self.accepted_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        self.n_accepted += 1
        return row

    def route(self, res: ExtractionResult, review_fields: list[str] | None = None) -> str:
        """Write to accepted if valid, otherwise quarantine. Returns the destination name."""
        model, errors = validate_record(res.doc_type, res.record_payload())
        if model is not None:
            self.write_accepted(res, review_fields)
            return "accepted"
        with self.quarantine_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"doc_id": res.doc_id, "extractor": res.extractor, "errors": errors, "raw": res.record_payload()},
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
        self.n_quarantined += 1
        return "quarantine"
