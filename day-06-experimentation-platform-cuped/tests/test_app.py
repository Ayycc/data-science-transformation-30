import pytest

st = pytest.importorskip("streamlit")


def test_app_renders_on_sample():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=120).run()
    assert not at.exception
    assert any("SRM chi-square" in m.value for m in at.markdown)
