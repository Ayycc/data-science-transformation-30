import numpy as np
import pandas as pd
from scipy import stats

from invopt.forecasting import lead_time_demand, pinball, quantile
from invopt.segmentation import abc_classes, sbc_class, xyz_classes


def test_abc_pareto():
    rev = pd.Series([700, 150, 100, 30, 20], index=list("abcde"))
    cls = abc_classes(rev, (0.8, 0.95))
    assert cls.to_dict() == {"a": "A", "b": "A", "c": "B", "d": "C", "e": "C"}


def test_xyz_cutoffs():
    assert list(xyz_classes(pd.Series([0.2, 0.7, 1.5]))) == ["X", "Y", "Z"]


def test_sbc_classes():
    assert sbc_class(np.array([5, 6, 5, 7, 6, 5])) == "smooth"
    assert sbc_class(np.array([1, 30, 2, 40, 1, 25])) == "erratic"
    assert sbc_class(np.array([0, 0, 3, 0, 0, 3, 0, 4])) == "intermittent"
    assert sbc_class(np.array([0, 0, 1, 0, 0, 40, 0, 2])) == "lumpy"


def test_lead_time_demand_formula():
    m, s = lead_time_demand(np.array([10.0]), np.array([3.0]), np.array([2.0]), np.array([0.5]), 1.0)
    assert m[0] == 30.0
    assert np.isclose(s[0], np.sqrt(3 * 9 + 100 * 0.25))


def test_quantiles():
    q = quantile(np.array([10.0, 10.0]), np.array([2.0, 2.0]), 0.95, np.array(["normal", "gamma"]))
    assert np.isclose(q[0], stats.norm.ppf(0.95, 10, 2))
    assert np.isclose(q[1], stats.gamma.ppf(0.95, a=25, scale=0.4))
    assert quantile(np.array([0.0]), np.array([1.0]), 0.9, np.array(["gamma"]))[0] == 0


def test_pinball():
    assert pinball(np.array([10.0]), np.array([8.0]), 0.9) == 0.9 * 2
    assert np.isclose(pinball(np.array([10.0]), np.array([12.0]), 0.9), 0.1 * 2)
