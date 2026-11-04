"""A query-answering server with budget enforcement (stretch goal).

Analysts send count or mean queries with a filter. Each answer is charged to the server's accountant *before*
noise is drawn; once the budget is gone every new query is refused. Repeating an identical query returns the
cached noisy answer at no extra cost (post-processing), which also stops averaging attacks that repeat a
query to cancel the noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .accounting import BudgetExceededError, PrivacyAccountant
from .data import CATEGORICAL, NUMERIC_BOUNDS


@dataclass
class Answer:
    query: str
    value: float | None
    epsilon_charged: float
    cached: bool = False
    refused: bool = False
    reason: str = ""


@dataclass
class QueryServer:
    df: pd.DataFrame
    accountant: PrivacyAccountant
    seed: int = 0
    log: list[Answer] = field(default_factory=list)
    _cache: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _mask(self, where: dict[str, str]) -> np.ndarray:
        m = np.ones(len(self.df), dtype=bool)
        for c, v in sorted(where.items()):
            if c not in CATEGORICAL or v not in CATEGORICAL[c]:
                raise ValueError(f"filters must use public categorical levels: {c}={v}")
            m &= (self.df[c] == v).to_numpy()
        return m

    def _answer(self, key: tuple, text: str, epsilon: float, compute) -> Answer:
        if key in self._cache:
            a = Answer(text, self._cache[key], 0.0, cached=True)
        else:
            try:
                self.accountant.spend(text, epsilon)
            except BudgetExceededError as exc:
                a = Answer(text, None, 0.0, refused=True, reason=str(exc))
                self.log.append(a)
                return a
            val = compute()
            self._cache[key] = val
            a = Answer(text, val, epsilon)
        self.log.append(a)
        return a

    def count(self, where: dict[str, str], epsilon: float) -> Answer:
        self._mask(where)  # validate the filter before anything is charged
        key = ("count", tuple(sorted(where.items())), epsilon)
        text = f"COUNT WHERE {where} @eps={epsilon}"
        return self._answer(key, text, epsilon, lambda: float(max(0.0, self._mask(where).sum() + self._rng.laplace(0, 1 / epsilon))))

    def mean(self, column: str, where: dict[str, str], epsilon: float) -> Answer:
        """Clipped mean: half the epsilon on the noisy sum, half on the noisy count."""
        if column not in NUMERIC_BOUNDS:
            raise ValueError(f"mean needs a numeric column with public bounds, got {column}")
        lo, hi = NUMERIC_BOUNDS[column]
        self._mask(where)
        key = ("mean", column, tuple(sorted(where.items())), epsilon)
        text = f"MEAN({column}) WHERE {where} @eps={epsilon}"

        def compute() -> float:
            v = self.df.loc[self._mask(where), column].clip(lo, hi).to_numpy(dtype=float)
            s = v.sum() + self._rng.laplace(0, max(abs(lo), abs(hi)) / (epsilon / 2))
            n = len(v) + self._rng.laplace(0, 1 / (epsilon / 2))
            return float(np.clip(s / max(n, 1.0), lo, hi))

        return self._answer(key, text, epsilon, compute)

    def summary(self) -> dict:
        return {
            "queries": len(self.log),
            "answered_new": sum(1 for a in self.log if not a.cached and not a.refused),
            "answered_from_cache": sum(1 for a in self.log if a.cached),
            "refused": sum(1 for a in self.log if a.refused),
            "budget": self.accountant.to_dict() | {"ledger": f"{len(self.accountant.ledger)} entries"},
        }
