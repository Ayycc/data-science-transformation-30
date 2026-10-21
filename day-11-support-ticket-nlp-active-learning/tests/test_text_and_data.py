import pandas as pd

from ticket_intel.config import SAMPLE_DIR
from ticket_intel.synth import TEMPLATES, add_typos, generate_base
from ticket_intel.text import normalise, strip_boilerplate


def test_generator_deterministic_and_15_intents():
    a, b = generate_base(10, seed=1), generate_base(10, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert a.intent.nunique() == 15 == len(TEMPLATES)


def test_typos_change_text_but_keep_length_close():
    import numpy as np

    s = "the password reset email never arrives in my inbox"
    t = add_typos(s, np.random.default_rng(0), rate=0.2)
    assert t != s and abs(len(t) - len(s)) < 10


def test_normalise_placeholders():
    out = normalise("Hi [NAME], order #123456 refund $49.99 for 3 seats!!!")
    assert out == "hi __name__ , order __id__ refund __amount__ for __num__ seats!"


def test_strip_boilerplate():
    t = normalise("Hi, this is [NAME]. i need to cancel order #12345 You can reach me at [EMAIL].\n\nThanks,\n[NAME]")
    assert strip_boilerplate(t) == "need to cancel order __id__"


def test_sample_files_small():
    for p in SAMPLE_DIR.iterdir():
        assert p.stat().st_size <= 1_000_000, p
    df = pd.read_csv(SAMPLE_DIR / "tickets_sample.csv")
    assert {"text", "intent", "pii_spans"} <= set(df.columns)
