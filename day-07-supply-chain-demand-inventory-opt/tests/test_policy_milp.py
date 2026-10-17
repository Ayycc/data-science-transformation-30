import numpy as np
from scipy import stats

from invopt.milp import Instance, greedy, objective_value, solve_ortools, solve_pulp
from invopt.policy import critical_ratio, round_to_pack


def test_round_to_pack():
    out = round_to_pack(np.array([0, 1, 13, 30]), np.array([6, 6, 6, 6]), np.array([12, 12, 12, 12]))
    assert list(out) == [0, 12, 18, 30]


def test_critical_ratio_newsvendor():
    cr = critical_ratio(np.array([20.0]), np.array([10.0]), np.array([4.0]), np.array([5.0]))
    assert np.isclose(cr[0], 15 / 21)
    # lognormal quantile at the critical ratio exceeds the median when CR > 0.5
    assert np.exp(np.log(100) + 0.3 * stats.norm.ppf(cr[0])) > 100


def _instance(budget=1e9, cap=(1e9, 1e9)):
    rng = np.random.default_rng(0)
    n = 14
    pack = rng.choice([1, 6, 12], n)
    moq = pack * rng.choice([1, 2], n)
    gap_S = rng.integers(5, 80, n).astype(float)
    return Instance(
        gap_s=gap_S / 3,
        gap_S=gap_S,
        pack=pack,
        moq=moq,
        cost=rng.uniform(2, 40, n),
        vol=rng.uniform(0.001, 0.02, n),
        wh=np.arange(n) % 2,
        w_s=rng.uniform(5, 30, n),
        w_S=rng.uniform(0.2, 2, n),
        h=rng.uniform(0.01, 0.2, n),
        K=30.0,
        budget=budget,
        cap_left=np.array(cap, float),
    )


def test_milp_respects_moq_budget_capacity_and_matches_pulp():
    inst = _instance(budget=3000.0, cap=(4.0, 3.0))
    sol = solve_ortools(inst, 10, 0.0)
    assert sol.status == "OPTIMAL" and sol.gap < 1e-6
    q = sol.q
    assert np.all((q == 0) | (q >= inst.moq))
    assert np.all(q % inst.pack == 0)
    assert (q * inst.cost).sum() <= inst.budget + 1e-6
    for w in (0, 1):
        assert (q * inst.vol)[inst.wh == w].sum() <= inst.cap_left[w] + 1e-9
    pl = solve_pulp(inst, 10)
    assert pl.status == "OPTIMAL"
    assert np.isclose(objective_value(inst, pl.q), sol.objective, rtol=1e-6)
    assert np.isclose(objective_value(inst, q), sol.objective, rtol=1e-6)
    g = greedy(inst)
    assert (g * inst.cost).sum() <= inst.budget + 1e-6
    assert objective_value(inst, g) >= sol.objective - 1e-6


def test_milp_unconstrained_is_separable_optimum():
    """With budget and capacity slack the MILP must equal the per-item brute-force optimum."""
    inst = _instance()
    sol = solve_ortools(inst, 10, 0.0)
    ub = inst.upper_packs()
    best = 0.0
    for i in range(inst.n):
        costs = []
        for n in range(ub[i] + 1):
            q = n * inst.pack[i]
            if 0 < q < inst.moq[i]:
                continue
            costs.append(
                inst.K * (q > 0)
                + inst.w_s[i] * max(inst.gap_s[i] - q, 0)
                + inst.w_S[i] * max(inst.gap_S[i] - q, 0)
                + inst.h[i] * q
            )
        best += min(costs)
    assert np.isclose(sol.objective, best, rtol=1e-6)
