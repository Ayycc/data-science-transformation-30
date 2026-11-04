"""Privacy budget accounting.

``PrivacyAccountant`` tracks (epsilon, delta) under basic sequential composition and *refuses* any spend that
would take the total over the configured budget: the check happens before noise is drawn, so an over-budget
query never produces an answer. ``ZCDPAccountant`` tracks zero-concentrated DP (rho), which composes additively
and converts to (epsilon, delta) far more tightly than basic composition when many Gaussian queries are answered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_TOL = 1e-12


class BudgetExceededError(RuntimeError):
    """Raised when a spend would take the cumulative budget above the configured total."""


@dataclass
class Spend:
    label: str
    epsilon: float
    delta: float = 0.0


@dataclass
class PrivacyAccountant:
    """(epsilon, delta) budget with basic sequential composition and a ledger of every spend."""

    epsilon_total: float
    delta_total: float = 0.0
    ledger: list[Spend] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.epsilon_total <= 0 or self.delta_total < 0:
            raise ValueError("epsilon_total must be > 0 and delta_total >= 0")

    @property
    def spent_epsilon(self) -> float:
        return math.fsum(s.epsilon for s in self.ledger)

    @property
    def spent_delta(self) -> float:
        return math.fsum(s.delta for s in self.ledger)

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.epsilon_total - self.spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        return max(0.0, self.delta_total - self.spent_delta)

    def can_spend(self, epsilon: float, delta: float = 0.0) -> bool:
        return self.spent_epsilon + epsilon <= self.epsilon_total + _TOL and self.spent_delta + delta <= self.delta_total + _TOL

    def spend(self, label: str, epsilon: float, delta: float = 0.0) -> Spend:
        if epsilon <= 0 or delta < 0 or not math.isfinite(epsilon):
            raise ValueError(f"invalid spend epsilon={epsilon}, delta={delta}")
        if not self.can_spend(epsilon, delta):
            raise BudgetExceededError(
                f"'{label}' asks for eps={epsilon:.4g}, delta={delta:.3g}; remaining eps={self.remaining_epsilon:.4g}, delta={self.remaining_delta:.3g}"
            )
        s = Spend(label, float(epsilon), float(delta))
        self.ledger.append(s)
        return s

    def sub_budget(self, label: str, epsilon: float, delta: float = 0.0) -> PrivacyAccountant:
        """Reserve part of the budget for a component, which then gets its own (nested) accountant."""
        self.spend(label, epsilon, delta)
        return PrivacyAccountant(epsilon, delta)

    def to_dict(self) -> dict:
        return {
            "epsilon_total": self.epsilon_total,
            "delta_total": self.delta_total,
            "epsilon_spent": round(self.spent_epsilon, 10),
            "delta_spent": self.spent_delta,
            "ledger": [{"label": s.label, "epsilon": s.epsilon, "delta": s.delta} for s in self.ledger],
        }


def zcdp_to_dp(rho: float, delta: float) -> float:
    """Smallest epsilon such that rho-zCDP implies (epsilon, delta)-DP (Canonne-Kamath-Steinke 2020, Prop. 12 style bound).

    Uses the tight conversion eps = min_alpha [ rho*alpha + log(1/delta)/(alpha-1) + log(1-1/alpha) ]
    evaluated on a fine grid of alpha; never looser than the classic rho + 2*sqrt(rho*log(1/delta)).
    """
    if rho <= 0:
        return 0.0
    classic = rho + 2 * math.sqrt(rho * math.log(1 / delta))
    best = classic
    for i in range(1, 4000):
        alpha = 1.0 + i * 0.01 * (1 + i / 200)
        eps = rho * alpha + (math.log(1 / delta) + (alpha - 1) * math.log(1 - 1 / alpha) - math.log(alpha)) / (alpha - 1)
        best = min(best, eps)
    return best


def dp_to_zcdp_rho(epsilon: float, delta: float) -> float:
    """Largest rho whose (epsilon, delta) conversion fits in ``epsilon`` (bisection on ``zcdp_to_dp``)."""
    lo, hi = 0.0, epsilon
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if zcdp_to_dp(mid, delta) <= epsilon else (lo, mid)
    return lo


@dataclass
class ZCDPAccountant:
    """rho-zCDP budget. Gaussian mechanism with L2 sensitivity s and noise sigma costs rho = s^2 / (2 sigma^2)."""

    rho_total: float
    ledger: list[tuple[str, float]] = field(default_factory=list)

    @classmethod
    def from_dp(cls, epsilon: float, delta: float) -> ZCDPAccountant:
        return cls(dp_to_zcdp_rho(epsilon, delta))

    @property
    def spent(self) -> float:
        return math.fsum(r for _, r in self.ledger)

    def spend(self, label: str, rho: float) -> None:
        if rho <= 0:
            raise ValueError("rho must be > 0")
        if self.spent + rho > self.rho_total + _TOL:
            raise BudgetExceededError(f"'{label}' asks for rho={rho:.4g}; remaining {self.rho_total - self.spent:.4g}")
        self.ledger.append((label, float(rho)))

    def epsilon(self, delta: float) -> float:
        return zcdp_to_dp(self.spent, delta)
