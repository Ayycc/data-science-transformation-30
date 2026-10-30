import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_regression import SNAP, current  # noqa: E402

SNAP.parent.mkdir(exist_ok=True)
SNAP.write_text(json.dumps(current(), indent=1, sort_keys=True) + "\n")
print("wrote", SNAP)
