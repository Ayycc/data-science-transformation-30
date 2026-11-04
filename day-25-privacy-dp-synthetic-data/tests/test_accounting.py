import math

import numpy as np
import pytest

from privacy25.accounting import BudgetExceededError, PrivacyAccountant, ZCDPAccountant, dp_to_zcdp_rho, zcdp_to_dp


def test_budget_never_exceeds_total_under_random_spends():
    rng = np.random.default_rng(0)
    for _trial in range(50):
        acc = PrivacyAccountant(epsilon_total=float(rng.uniform(0.5, 5)), delta_total=1e-5)
        for i in range(60):
            eps, delta = float(rng.exponential(0.3)) + 1e-6, float(rng.choice([0.0, 1e-7, 1e-6]))
            before = list(acc.ledger)
            try:
                acc.spend(f"q{i}", eps, delta)
            except BudgetExceededError:
                assert acc.ledger == before  # a refused spend leaves no trace
            assert acc.spent_epsilon <= acc.epsilon_total + 1e-12
            assert acc.spent_delta <= acc.delta_total + 1e-18


def test_exact_budget_is_allowed_and_one_more_is_refused():
    acc = PrivacyAccountant(1.0)
    for _ in range(10):
        acc.spend("tenth", 0.1)
    assert math.isclose(acc.spent_epsilon, 1.0)
    with pytest.raises(BudgetExceededError):
        acc.spend("one too many", 1e-6)


def test_sub_budget_is_charged_up_front_and_enforced():
    acc = PrivacyAccountant(2.0)
    sub = acc.sub_budget("component", 1.5)
    assert acc.remaining_epsilon == pytest.approx(0.5)
    sub.spend("a", 1.0)
    with pytest.raises(BudgetExceededError):
        sub.spend("b", 0.6)
    with pytest.raises(BudgetExceededError):
        acc.sub_budget("too big", 0.6)


def test_invalid_spends_rejected():
    acc = PrivacyAccountant(1.0)
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError):
            acc.spend("bad", bad)


def test_zcdp_conversion_is_tight_and_invertible():
    for rho in (0.001, 0.05, 0.5, 2.0):
        classic = rho + 2 * math.sqrt(rho * math.log(1e6))
        assert zcdp_to_dp(rho, 1e-6) <= classic + 1e-12
    rho = dp_to_zcdp_rho(1.0, 1e-6)
    assert zcdp_to_dp(rho, 1e-6) <= 1.0
    assert zcdp_to_dp(rho * 1.01, 1e-6) > 1.0 - 1e-3
    z = ZCDPAccountant.from_dp(1.0, 1e-6)
    z.spend("half", z.rho_total / 2)
    z.spend("half", z.rho_total / 2)
    with pytest.raises(BudgetExceededError):
        z.spend("more", 1e-6)
    assert z.epsilon(1e-6) <= 1.0 + 1e-9
