"""make run: regenerate the SYNTHETIC corpus, extract, validate, evaluate, write results/."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract_eval.pipeline import main  # noqa: E402

if __name__ == "__main__":
    m = main()
    print(
        json.dumps(
            {
                k: {
                    "field_acc": round(v["field_accuracy_fuzzy"], 4),
                    "valid": round(v["schema_valid_rate"], 4),
                    "auto_accept": round(v["routing_test"]["auto_accept_rate"], 4),
                }
                for k, v in m["extractors"].items()
            },
            indent=1,
        )
    )
    print("llm:", m["llm"]["status"], "| runtime s:", round(m["runtime"]["seconds"], 1))
