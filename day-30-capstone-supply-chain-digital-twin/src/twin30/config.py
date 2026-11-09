"""Network, product and economics configuration for the SYNTHETIC retail supply chain.

Everything here is invented for the capstone. The "world" generator uses these values (plus hidden
truth parameters in ``world.py``) to write the historical record; the digital twin only reads the
operational and financial constants (prices, costs, distances, policy rules), which a real retailer
would know from its own systems, and learns demand, promo effects, lead times and disruption rates
from the history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SEED = 20261109
DAYS_PER_YEAR = 364  # 52 whole weeks keeps weekday alignment exact
HISTORY_DAYS = 2 * DAYS_PER_YEAR
HISTORY_START = "2024-11-04"  # a Monday; history ends 2026-11-01, the plan year starts 2026-11-02


@dataclass(frozen=True)
class Category:
    name: str
    price: float  # regular shelf price, USD
    unit_cost: float  # standard cost from the primary supplier, USD
    promo_discount: float  # share off the shelf price during a promo week
    promo_weeks_per_year: int
    freight_per_unit_km: float  # outbound DC->store cost, USD per unit per km


CATEGORIES: tuple[Category, ...] = (
    Category("staples", 4.0, 2.60, 0.10, 10, 0.00100),
    Category("household", 9.0, 5.40, 0.10, 8, 0.00150),
    Category("seasonal", 25.0, 13.0, 0.15, 6, 0.00300),
)


@dataclass(frozen=True)
class Site:
    name: str
    x: float  # km on a 1,000 x 700 km synthetic map
    y: float


STORES: tuple[Site, ...] = (
    Site("N1", 180, 560),
    Site("N2", 260, 610),
    Site("N3", 120, 500),
    Site("N4", 320, 540),
    Site("S1", 200, 140),
    Site("S2", 280, 90),
    Site("S3", 140, 200),
    Site("S4", 340, 160),
    Site("E1", 820, 360),
    Site("E2", 900, 420),
    Site("E3", 860, 280),
    Site("E4", 760, 440),
)

DCS: tuple[Site, ...] = (Site("DC-North", 230, 520), Site("DC-South", 250, 170), Site("DC-East", 830, 370))
EXISTING_DCS = (0, 1)  # DC-East is the candidate site; it does not exist in the history

# store -> DC assignment used throughout the history (East stores split by straight-line distance)
HISTORICAL_ASSIGNMENT = (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0)


@dataclass(frozen=True)
class Supplier:
    name: str
    cost_factor: float  # multiplier on the category standard cost (landed)
    lead_mean_days: float  # TRUE mean lead time (hidden from the twin)
    lead_cv: float
    outage_rate_per_year: float  # TRUE Poisson rate (hidden from the twin)
    outage_mean_days: float


SUPPLIERS: tuple[Supplier, ...] = (
    Supplier("S-Offshore", 1.00, 24.0, 0.22, 2.0, 18.0),
    Supplier("S-Nearshore", 1.07, 6.0, 0.30, 0.5, 6.0),
)


@dataclass(frozen=True)
class Finance:
    holding_rate_per_year: float = 0.25  # of inventory value at standard cost
    dc_handling_per_unit: float = 0.05
    dc_fixed_cost_per_year: tuple[float, float, float] = (1_800_000.0, 1_800_000.0, 1_200_000.0)
    dc_capacity_units_per_day: tuple[float, float, float] = (14_000.0, 12_000.0, 9_000.0)
    km_per_transit_day: float = 450.0


FINANCE = Finance()


@dataclass(frozen=True)
class Policy:
    """Operational replenishment rule used by the retailer (known, so shared by world and twin)."""

    store_z: float = 1.5  # safety factor on store order-up-to levels
    dc_z: float = 1.5  # safety factor on DC base-stock levels
    nearshore_share: float = 0.10  # share of each weekly DC order placed with the nearshore supplier
    forecast_window: int = 28  # trailing days used by the legacy store forecast
    planned_promo_lift: float = 1.40  # lift the planners assume when a promo is in the cover window
    dc_review_days: int = 7  # DCs order from suppliers every Monday
    max_cover_days: float = 4.5  # store shelf + backroom space: order-up-to never exceeds this many days of sales


BASELINE_POLICY = Policy()


@dataclass(frozen=True)
class Decisions:
    """Strategic levers evaluated by the scenario engine."""

    open_dc_east: bool = False
    assignment: tuple[int, ...] = HISTORICAL_ASSIGNMENT
    nearshore_share: float = 0.10
    store_z: float = 1.5
    dc_z: float = 1.5
    promo_strategy: str = "current"  # current | targeted | none

    def policy(self) -> Policy:
        return Policy(store_z=self.store_z, dc_z=self.dc_z, nearshore_share=self.nearshore_share)


@dataclass(frozen=True)
class Geometry:
    dist: np.ndarray = field(repr=False)  # (n_dc, n_store) km
    transit: np.ndarray = field(repr=False)  # (n_dc, n_store) nominal days


def geometry() -> Geometry:
    d = np.array([[np.hypot(dc.x - s.x, dc.y - s.y) for s in STORES] for dc in DCS])
    transit = 1 + np.floor(d / FINANCE.km_per_transit_day).astype(int)
    return Geometry(dist=d, transit=transit)


N_STORES = len(STORES)
N_CATS = len(CATEGORIES)
N_DCS = len(DCS)
N_SUPPLIERS = len(SUPPLIERS)
PRICE = np.array([c.price for c in CATEGORIES])
UNIT_COST = np.array([c.unit_cost for c in CATEGORIES])
DISCOUNT = np.array([c.promo_discount for c in CATEGORIES])
FREIGHT = np.array([c.freight_per_unit_km for c in CATEGORIES])
SUPPLIER_COST = np.array([s.cost_factor for s in SUPPLIERS])
