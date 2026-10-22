"""Entry point for `make run`: regenerate SYNTHETIC data, run every analysis, write results/."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from policy_causal.config import RunConfig, quick_config  # noqa: E402
from policy_causal.pipeline import run  # noqa: E402

if __name__ == "__main__":
    cfg = quick_config() if os.environ.get("DAY12_QUICK") == "1" else RunConfig()
    m = run(cfg, lalonde_source=os.environ.get("LALONDE_SOURCE", "auto"))
    a = m["acceptance"]
    print(
        f"earnings ATE {a['earnings_ate']:.0f} CI {a['earnings_ci'][0]:.0f}..{a['earnings_ci'][1]:.0f} (truth {m['truth']['ate_earnings']:.0f}) in CI: {a['earnings_truth_in_ci']}"
    )
    print(f"employment ATE {a['employment_ate']:.4f} (truth {m['truth']['ate_employment']:.4f}) in CI: {a['employment_truth_in_ci']}")
    print(
        f"refutations passed {a['refutation_tests_passed']}/{a['refutation_tests_reported']}; LaLonde path: {m['data']['lalonde_path']}; runtime {m['runtime']['seconds_total']} s"
    )
