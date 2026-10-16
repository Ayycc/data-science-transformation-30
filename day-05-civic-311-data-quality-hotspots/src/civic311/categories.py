"""Canonical service-request categories, the raw-spelling catalog, and fuzzy harmonisation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .config import CATALOG_SEED


@dataclass(frozen=True)
class Category:
    name: str
    department: str
    share: float  # share of annual volume
    median_hours: float  # median time to close
    sigma: float  # lognormal spread of time to close
    season_amp: float  # seasonal amplitude of volume
    season_peak_doy: int  # day of year with peak volume
    abbrev: tuple[str, ...]  # abbreviations used by legacy intake systems
    legacy_code: str  # old CRM type code
    synonyms: tuple[str, ...]  # free-text spellings seen in other intake channels


CATEGORIES: tuple[Category, ...] = (
    Category("Pothole in Street", "CDOT", 0.12, 96, 0.9, 0.6, 80, ("Pothole in St", "Pot Hole In Street"), "PHF",
             ("Street Pothole", "Hole in road")),
    Category("Street Light Out", "CDOT", 0.09, 72, 0.8, 0.3, 340, ("St Light Out", "Street Lt Out"), "SLA",
             ("Streetlight Out", "Light Out - Street")),
    Category("Alley Light Out", "CDOT", 0.04, 96, 0.8, 0.3, 340, ("Alley Lt Out",), "SFA", ("Light Out - Alley",)),
    Category("Graffiti Removal", "Streets & Sanitation", 0.08, 36, 0.9, 0.2, 200, ("Graffiti Rmvl",), "GRAF",
             ("Remove Graffiti", "Graffiti Complaint")),
    Category("Rodent Baiting", "Streets & Sanitation", 0.07, 120, 0.8, 0.5, 190, ("Rodent Bait",), "SGA",
             ("Rat Problem", "Rodent Complaint")),
    Category("Abandoned Vehicle", "Police", 0.06, 240, 0.9, 0.1, 150, ("Abandoned Veh", "Abnd Vehicle"), "SKA",
             ("Vehicle Abandoned",)),
    Category("Garbage Cart Maintenance", "Streets & Sanitation", 0.08, 60, 0.7, 0.1, 150,
             ("Garbage Cart Maint",), "SIE", ("Broken Garbage Cart", "Garbage Cart Repair")),
    Category("Tree Trim", "Streets & Sanitation", 0.06, 480, 0.9, 0.6, 180, ("Tree Trim Req",), "SEC",
             ("Trim Tree", "Tree Trimming")),
    Category("Water On Street", "Water Management", 0.05, 48, 0.8, 0.3, 120, ("Water On St",), "WBJ",
             ("Street Flooding",)),
    Category("Sewer Cave-In Inspection", "Water Management", 0.03, 168, 0.8, 0.3, 120,
             ("Sewer Cave In Insp",), "AAE", ("Sewer Cave In",)),
    Category("Traffic Signal Out", "CDOT", 0.04, 12, 0.7, 0.1, 200, ("Traffic Sig Out",), "SFC",
             ("Signal Out", "Traffic Light Out")),
    Category("Sidewalk Inspection", "CDOT", 0.05, 360, 0.8, 0.3, 140, ("Sidewalk Insp",), "SDW",
             ("Broken Sidewalk",)),
    Category("Building Violation", "Buildings", 0.07, 300, 0.9, 0.1, 150, ("Bldg Violation",), "BBA",
             ("Building Code Complaint",)),
    Category("Noise Complaint", "Police", 0.07, 24, 1.0, 0.5, 200, ("Noise Compl",), "NOI", ("Loud Noise",)),
    Category("Dead Animal Pick-Up", "Streets & Sanitation", 0.09, 18, 0.7, 0.3, 200, ("Dead Animal Pickup",),
             "SGQ", ("Dead Animal", "Animal Carcass")),
)  # fmt: skip
CANONICAL = tuple(c.name for c in CATEGORIES)
BY_NAME = {c.name: c for c in CATEGORIES}

# Junk values with no recoverable category (they must be quarantined, never guessed).
JUNK_VALUES = ("TEST ENTRY", "Other", "Misc. Request", "N/A", "General Inquiry", "see notes", "???")

_ABBREV = {
    "st": "street", "lt": "light", "rmvl": "removal", "veh": "vehicle", "abnd": "abandoned", "maint": "maintenance",
    "req": "", "insp": "inspection", "sig": "signal", "bldg": "building", "compl": "complaint", "pickup": "pick up",
    "pot": "pot", "cave": "cave",
}  # fmt: skip


def normalise(s: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace and expand common abbreviations."""
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [_ABBREV.get(t, t) for t in s.split()]
    return " ".join(t for t in toks if t)


_NORM_CANON = {normalise(c): c for c in CANONICAL}


def _typo(s: str, rng: np.random.Generator) -> str:
    i = int(rng.integers(1, len(s) - 2))
    op = rng.integers(3)
    if op == 0:  # drop a character
        return s[:i] + s[i + 1 :]
    if op == 1:  # swap neighbours
        return s[:i] + s[i + 1] + s[i] + s[i + 2 :]
    return s[:i] + s[i] + s[i:]  # double a character


def variant_catalog() -> pd.DataFrame:
    """Every raw spelling the generator can emit for each category (fixed seed, independent of data seed).

    kind: format (case/whitespace), typo, abbrev, legacy_code, synonym.
    """
    rng = np.random.default_rng(CATALOG_SEED)
    rows = []
    for c in CATEGORIES:
        forms = {
            (c.name.upper(), "format"),
            (c.name.lower(), "format"),
            (c.name + "  ", "format"),
            ("  " + c.name.replace(" ", "  "), "format"),
            (_typo(c.name, rng), "typo"),
            (_typo(c.name, rng), "typo"),
            (_typo(c.name.lower(), rng), "typo"),
            (c.legacy_code, "legacy_code"),
        }
        forms |= {(a, "abbrev") for a in c.abbrev} | {(a.upper(), "abbrev") for a in c.abbrev[:1]}
        forms |= {(s, "synonym") for s in c.synonyms}
        for raw, kind in sorted(forms):
            if raw != c.name:
                rows.append({"raw_value": raw, "true_category": c.name, "kind": kind})
    for j in JUNK_VALUES:
        rows.append({"raw_value": j, "true_category": None, "kind": "junk"})
    return pd.DataFrame(rows).drop_duplicates("raw_value", keep="first").reset_index(drop=True)


# ---------------------------------------------------------------- harmonisation
AUTO_ACCEPT = 90.0  # fuzzy score at or above which a match is accepted without review
REVIEW_FLOOR = 70.0  # below this the value is proposed as unmapped


def propose_mapping(raw_values: pd.Series | list[str]) -> pd.DataFrame:
    """Propose a canonical category for each distinct raw value.

    method: exact | normalised | fuzzy_auto (score >= AUTO_ACCEPT) | needs_review | no_match | missing
    """
    vals = pd.Series(pd.unique(pd.Series(raw_values, dtype=object).dropna())).astype(str)
    norm_keys = list(_NORM_CANON)
    out = []
    for v in vals:
        if v in BY_NAME:
            out.append((v, v, 100.0, "exact"))
            continue
        n = normalise(v)
        if n in _NORM_CANON:
            out.append((v, _NORM_CANON[n], 100.0, "normalised"))
            continue
        if not n:
            out.append((v, None, 0.0, "no_match"))
            continue
        best, score, _ = process.extractOne(n, norm_keys, scorer=fuzz.WRatio)
        cand = _NORM_CANON[best]
        method = "fuzzy_auto" if score >= AUTO_ACCEPT else "needs_review" if score >= REVIEW_FLOOR else "no_match"
        out.append((v, cand, round(float(score), 1), method))
    return pd.DataFrame(out, columns=["raw_value", "proposed_category", "score", "method"])


def load_reviewed(path: Path) -> pd.DataFrame:
    """The human-reviewed mapping table: raw_value -> canonical_category (empty = reject)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    bad = set(df.loc[df.canonical_category != "", "canonical_category"]) - set(CANONICAL)
    if bad:
        raise ValueError(f"reviewed mapping points at unknown categories: {sorted(bad)}")
    if df.raw_value.duplicated().any():
        raise ValueError("reviewed mapping has duplicate raw values")
    return df


def build_mapping(raw_values: pd.Series, reviewed: pd.DataFrame | None) -> pd.DataFrame:
    """Final mapping table for the distinct raw values: proposals, overridden by reviewed decisions.

    Values that still need review (no reviewed decision) map to None and are quarantined as CATEGORY_UNMAPPED.
    """
    m = propose_mapping(raw_values)
    m["category"] = np.where(m.method.isin(["exact", "normalised", "fuzzy_auto"]), m.proposed_category, None)
    m["source"] = np.where(m.category.notna(), "auto", "unresolved")
    if reviewed is not None and len(reviewed):
        r = reviewed.set_index("raw_value")
        hit = m.raw_value.isin(r.index)
        decided = r.loc[m.loc[hit, "raw_value"], "canonical_category"].replace("", None).to_numpy()
        m.loc[hit, "category"] = decided
        m.loc[hit, "source"] = "reviewed"
    return m
