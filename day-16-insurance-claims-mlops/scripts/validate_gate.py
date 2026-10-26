"""CI model-validation gate.

Builds a throwaway local MLflow registry, registers the GLM baseline as the current champion, trains the
GBM challenger from the code under test and applies the promotion gate. Exits 1 if the challenger
does not beat the champion, so a pull request that degrades the model fails CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claims_mlops import config as C  # noqa: E402
from claims_mlops import data as D  # noqa: E402
from claims_mlops import models as M  # noqa: E402
from claims_mlops import registry as R  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies-per-month", type=int, default=C.SimConfig().policies_per_month)
    ap.add_argument("--out", type=Path, default=None, help="optional JSON file for the decision")
    args = ap.parse_args()
    sim = C.SimConfig(n_months=12, policies_per_month=args.policies_per_month)
    df = D.generate_synthetic(sim)
    train, valid = D.split_months(df, range(1, 11)), D.split_months(df, [11, 12])
    with tempfile.TemporaryDirectory() as td:
        client = R.setup(Path(td))
        glm = M.fit(M.build_glm(), train)
        champ = R.log_candidate(glm, "glm", {"alpha": 1e-3}, train, valid)
        R.apply_decision(client, R.gate(champ, None, None, C.GatePolicy()))
        pipe, cat_idx = M.build_gbm()
        cand = R.log_candidate(M.fit(pipe, train, cat_idx), "gbm", {}, train, valid)
        ver, cm = R.evaluate_champion(client, valid)
        decision = R.gate(cand, cm, ver, C.GatePolicy())
        action = R.apply_decision(client, decision)
    summary = {"action": action, "checks": decision.checks}
    print(json.dumps(summary, indent=2, default=float))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2, default=float))
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
