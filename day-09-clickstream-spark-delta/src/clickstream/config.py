"""Scale presets. Every number in the README comes from the FULL preset."""

from __future__ import annotations

from dataclasses import dataclass, field

SEED = 20261019
BASE_DATE = "2026-10-01"  # first event day (UTC)
SESSION_GAP_S = 30 * 60  # 30-minute inactivity window


@dataclass(frozen=True)
class Scale:
    name: str
    n_users: int
    bot_frac: float
    n_articles: int
    days: int = 3
    mean_sessions_per_day: float = 1.25
    mean_events_per_session: float = 7.0
    bot_events_per_day: tuple[int, int] = (900, 2600)
    viral_share: float = 0.30  # share of human events that hit the 5 viral articles
    viral_weights: tuple[float, ...] = (0.60, 0.13, 0.10, 0.09, 0.08)
    skew_user_frac: float = 0.03  # devices with a wrong-timezone clock
    late_frac: float = 0.015  # events delivered in the next day's batch
    dup_frac: float = 0.02  # duplicate deliveries
    campaigns_per_viral_article: int = 120
    shuffle_partitions: int = 64
    seed: int = SEED
    extra: dict = field(default_factory=dict)


FULL = Scale(name="full", n_users=650_000, bot_frac=0.0005, n_articles=200_000)
# Tiny preset used by the tests and for the committed sample (same generator).
TEST = Scale(name="test", n_users=300, bot_frac=0.02, n_articles=300, bot_events_per_day=(150, 250), shuffle_partitions=4)

PRESETS = {"full": FULL, "test": TEST}
