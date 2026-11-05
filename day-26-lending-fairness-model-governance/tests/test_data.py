import numpy as np
import pytest

from credit26 import data as D


def test_generator_is_seeded_and_labelled(data):
    a = D.generate_applications(n=500, seed=1)
    b = D.generate_applications(n=500, seed=1)
    assert a.equals(b)
    df = data.df
    # recorded label = unbiased label plus spurious bads: never the other way round
    assert (df["bad"] >= df["bad_unbiased"]).all()
    rates = df.groupby("group")["bad"].mean() - df.groupby("group")["bad_unbiased"].mean()
    assert rates["B"] > rates["A"] and rates["C"] > rates["A"]
    # geography is a proxy: clusters >= 10 are mostly protected-group applicants
    assert (df.loc[df["zip_cluster"] >= 10, "group"] != "A").mean() > 0.6
    assert set(D.SYNTH_SPEC.features).isdisjoint({"group", "sex", "age", "age_band"})


def test_time_split(parts):
    assert parts.train["month"].max() <= 14 < parts.valid["month"].min()
    assert parts.test["month"].min() == 19


def test_fallback_records_attempts(monkeypatch):
    def boom(url, timeout=20.0):
        raise OSError("Tunnel connection failed: 403 Forbidden")

    monkeypatch.setattr(D, "_fetch", boom)
    cd = D.load_data("auto", n=300)
    assert cd.path_used == "synthetic_generator"
    assert [a["source"] for a in cd.attempts] == ["hmda", "german_credit", "synthetic_generator"]
    assert not cd.attempts[0]["ok"] and "403" in cd.attempts[0]["error"]
    with pytest.raises(OSError):
        D.load_data("hmda", n=300)


def test_parse_german_fixture():
    row_good = "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1"
    row_bad = "A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2"
    df = D.parse_german(row_good + "\n" + row_bad + "\n")
    assert list(df["bad"]) == [0, 1]
    assert list(df["sex"]) == ["male", "female"]
    assert list(df["age_band"]) == ["25+", "<25"]
    assert set(D.GERMAN_SPEC.features) <= set(df.columns)


def test_parse_hmda_fixture():
    csv = (
        "action_taken,loan_amount,income,debt_to_income_ratio,loan_to_value_ratio,property_value,loan_term,derived_race,derived_sex\n"
        "1,205000,85,20%-<30%,80,255000,360,White,Male\n"
        "3,105000,40,>60%,95,115000,360,Black or African American,Female\n"
        "3,155000,60,42,90,175000,360,Asian,Joint\n"
        "4,155000,60,42,90,175000,360,White,Male\n"
    )
    df = D.parse_hmda(csv)
    assert len(df) == 2  # 'Joint' sex and action 4 (withdrawn) dropped
    assert list(df["bad"]) == [0, 1]
    assert np.allclose(df["debt_to_income"], [0.25, 0.65])
