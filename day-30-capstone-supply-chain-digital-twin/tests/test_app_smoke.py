from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "results" / "model"


def test_backend_imports_and_streamlit_is_installed():
    import streamlit  # noqa: F401

    from twin30 import app_backend  # noqa: F401


@pytest.mark.skipif(not (MODEL / "plan_context.json").exists(), reason="results/model not built yet (run make run)")
def test_backend_evaluates_a_plan_from_committed_artefacts():
    from twin30 import app_backend

    ctx, meta = app_backend.load_context()
    dec = app_backend.decisions(meta, open_east=True, nearshore=0.25, store_z=2.0, promo="targeted")
    mine, base, s = app_backend.evaluate(ctx, dec, "normal", reps=3)
    assert len(mine) == 3 and set(["expected_profit", "cvar5_profit", "prob_beats_baseline"]) <= set(s)


@pytest.mark.skipif(not (MODEL / "plan_context.json").exists(), reason="results/model not built yet (run make run)")
def test_streamlit_app_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.metric) == 4
