"""Sample-ratio mismatch (SRM) detection with a chi-square goodness-of-fit test."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SRMResult:
    observed: tuple[int, ...]
    expected: tuple[float, ...]
    chi2: float
    p_value: float
    threshold: float
    flagged: bool

    @property
    def observed_share(self) -> tuple[float, ...]:
        total = sum(self.observed)
        return tuple(o / total for o in self.observed)


def srm_test(observed: Sequence[int], expected_ratios: Sequence[float], threshold: float = 0.001) -> SRMResult:
    """Chi-square test of observed arm counts against the designed allocation.

    The conventional alarm threshold is p < 0.001: SRM checks run on every experiment, so a
    strict threshold keeps false alarms rare while real assignment bugs give tiny p-values.
    """
    obs = np.asarray(observed, dtype=float)
    ratios = np.asarray(expected_ratios, dtype=float)
    if obs.shape != ratios.shape or obs.ndim != 1 or len(obs) < 2:
        raise ValueError("observed and expected_ratios must be 1-D and the same length (>= 2)")
    if np.any(ratios <= 0):
        raise ValueError("expected ratios must be positive")
    exp = ratios / ratios.sum() * obs.sum()
    chi2, p = stats.chisquare(obs, exp)
    return SRMResult(
        observed=tuple(int(o) for o in obs),
        expected=tuple(float(e) for e in exp),
        chi2=float(chi2),
        p_value=float(p),
        threshold=threshold,
        flagged=bool(p < threshold),
    )
