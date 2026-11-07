import pandas as pd
import pytest

from legacy28.config import ELR_BP
from legacy28.legacy import MISSING, excel_vlookup_approx, sas_add, sas_round_div, sas_sub, step01_policy_prep


def test_sas_round_half_away_from_zero():
    assert sas_round_div(5, 2) == 3  # 2.5 -> 3 (Python round() would give 2)
    assert sas_round_div(-5, 2) == -3
    assert sas_round_div(7, 4) == 2
    assert sas_round_div(1, 3) == 0


def test_sas_missing_arithmetic():
    assert sas_add(10, MISSING) is MISSING
    assert sas_sub(MISSING, 0) is MISSING
    assert sas_add(1, 2) == 3


def test_vlookup_approximate_match_quirk():
    assert excel_vlookup_approx("MOTOR", ELR_BP) == ELR_BP["MOTOR"]
    # OTHER is not in the table: approximate match silently returns MOTOR's row
    assert excel_vlookup_approx("OTHER", ELR_BP) == ELR_BP["MOTOR"]
    with pytest.raises(ValueError):
        excel_vlookup_approx("AAA", ELR_BP)


def test_nodupkey_depends_on_file_order():
    pv = pd.DataFrame(
        {
            "load_seq": [1, 2],
            "policy_id": [9, 9],
            "version_no": [2, 1],
            "version_eff": [200, 100],
            "lob_code_raw": ["HOME", "HOME"],
            "region": ["EAST", "WEST"],
            "load_date": [100, 100],
        }
    )
    assert step01_policy_prep(pv)[9][1] == "EAST"
    flipped = pv.assign(load_seq=[2, 1])
    assert step01_policy_prep(flipped)[9][1] == "WEST"  # same data, different file order, different answer


def test_lob_mapping_is_case_sensitive():
    pv = pd.DataFrame(
        {
            "load_seq": [1, 2],
            "policy_id": [1, 2],
            "version_no": [1, 1],
            "version_eff": [0, 0],
            "lob_code_raw": ["Motor", "home "],
            "region": ["EAST", "EAST"],
            "load_date": [0, 0],
        }
    )
    pol = step01_policy_prep(pv)
    assert pol[1][0] == "OTHER" and pol[2][0] == "OTHER"


def test_generator_plants_every_quirk(final):
    pv, cv = final.policy_versions, final.claim_valuations
    assert (pv.groupby("policy_id").size() > 1).any()
    assert (~pv["lob_code_raw"].isin(list(ELR_BP))).any()
    assert cv["case_reserve_cents"].isna().any()
    # backfill: some policy has a later version physically ahead of version 1
    first = pv.sort_values("load_seq").drop_duplicates("policy_id")
    assert (first["version_no"] > 1).any()


def test_legacy_loses_paid_when_reserve_missing(legacy_out):
    c = legacy_out["claims_detail"]
    miss = c[c["case_reserve_cents"].isna()]
    assert len(miss) > 0 and miss["incurred_cum_cents"].isna().all()
