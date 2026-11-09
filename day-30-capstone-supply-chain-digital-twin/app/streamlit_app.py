"""What-if explorer for the SYNTHETIC supply-chain digital twin.

Run:  make app   (streamlit run app/streamlit_app.py)
Reads results/model/ written by `make run`; every simulation runs live in the SimPy twin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin30 import app_backend  # noqa: E402

st.set_page_config(page_title="Digital twin what-if (SYNTHETIC)", layout="wide")
st.title("Supply-chain digital twin: what-if explorer")
st.caption(
    "SYNTHETIC retailer: 12 stores, 3 categories, 2 DCs (+ candidate DC-East), offshore + nearshore suppliers. Plan year starting 2026-11-02."
)


@st.cache_resource
def _ctx():
    return app_backend.load_context()


ctx, meta = _ctx()
opt = meta["optimised"]
with st.sidebar:
    st.header("Decisions")
    open_east = st.checkbox("Open DC-East", value=False)
    nearshore = st.slider("Nearshore supplier share", 0.0, 0.6, 0.10, 0.05)
    store_z = st.slider("Store safety factor z", 0.5, 5.0, 1.5, 0.25)
    promo = st.selectbox("Promo strategy", ["current", "targeted", "none"])
    st.header("World")
    regime = st.selectbox(
        "Disruption regime",
        ["normal", "stress", "mixed", "calm"],
        help="stress = calibrated outages + a forced 28-day offshore outage from plan day 14",
    )
    reps = st.slider("Monte Carlo replications", 10, 300, 30, 10)
    st.caption(
        f"Optimiser's choice: DC-East {'open' if opt['open_dc_east'] else 'closed'}, nearshore {opt['nearshore_share']:.0%}, z {opt['store_z']}, promos {opt['promo_strategy']}"
    )

dec = app_backend.decisions(meta, open_east, nearshore, store_z, promo)
with st.spinner(f"Simulating {2 * reps} plan years..."):
    mine, base, s = app_backend.evaluate(ctx, dec, regime, reps)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Expected operating profit", f"${s['expected_profit'] / 1e6:,.2f}M", f"{s['delta_profit_vs_baseline_mean'] / 1e6:+,.2f}M vs status quo"
)
c2.metric("CVaR 5% (worst-year profit)", f"${s['cvar5_profit'] / 1e6:,.2f}M", f"{s['delta_cvar5_vs_baseline'] / 1e6:+,.2f}M")
c3.metric("Fill rate (mean)", f"{s['fill_rate_mean']:.1%}", f"{(s['fill_rate_mean'] - base.fill_rate.mean()) * 100:+.1f} pp")
c4.metric("P(beats status quo)", f"{s['prob_beats_baseline']:.0%}")

st.subheader("Profit distribution (paired seeds)")
chart = (
    mine[["operating_profit"]].rename(columns={"operating_profit": "this plan"}).assign(status_quo=base.operating_profit.to_numpy()) / 1e6
)
st.bar_chart(chart.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).loc[["5%", "25%", "50%", "75%", "95%"]])
st.dataframe(
    mine[["operating_profit", "revenue", "fill_rate", "in_stock_rate", "avg_dc_inventory_units", "outbound_transport_cost"]]
    .describe()
    .T.round(3),
    width="stretch",
)
st.caption("All figures SYNTHETIC. Same random seeds are used for the plan and the status quo, so differences are not noise between runs.")
