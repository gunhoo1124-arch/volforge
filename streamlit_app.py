"""VolForge Streamlit entry point."""

import streamlit as st

st.set_page_config(
    page_title="VolForge",
    page_icon=":material/candlestick_chart:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page(
            "app_pages/pricing.py",
            title="Pricing calculator",
            icon=":material/calculate:",
            default=True,
        ),
        st.Page(
            "app_pages/implied_volatility.py",
            title="Implied volatility",
            icon=":material/query_stats:",
        ),
        st.Page(
            "app_pages/chain_explorer.py",
            title="Option-chain explorer",
            icon=":material/table_chart:",
        ),
        st.Page(
            "app_pages/data_quality.py",
            title="Data quality",
            icon=":material/fact_check:",
        ),
        st.Page(
            "app_pages/volatility_surfaces.py",
            title="Volatility surfaces",
            icon=":material/stacked_line_chart:",
        ),
        st.Page(
            "app_pages/heston_calibration.py",
            title="Heston calibration",
            icon=":material/model_training:",
        ),
        st.Page(
            "app_pages/numerical_engines.py",
            title="Numerical engines",
            icon=":material/function:",
        ),
        st.Page(
            "app_pages/american_options.py",
            title="American options",
            icon=":material/account_tree:",
        ),
        st.Page(
            "app_pages/heston_greeks.py",
            title="Heston Greeks",
            icon=":material/speed:",
        ),
        st.Page(
            "app_pages/hedging.py",
            title="Hedging laboratory",
            icon=":material/balance:",
        ),
        st.Page(
            "app_pages/hedge_robustness.py",
            title="Hedge robustness",
            icon=":material/verified_user:",
        ),
        st.Page(
            "app_pages/point_in_time_replay.py",
            title="Point-in-time replay",
            icon=":material/history:",
        ),
        st.Page(
            "app_pages/deliverable_valuation.py",
            title="Deliverable valuation",
            icon=":material/inventory_2:",
        ),
        st.Page(
            "app_pages/basket_pricing.py",
            title="Basket pricing",
            icon=":material/hub:",
        ),
        st.Page(
            "app_pages/basket_replay.py",
            title="Basket hedge replay",
            icon=":material/route:",
        ),
        st.Page(
            "app_pages/basket_inputs.py",
            title="Basket input store",
            icon=":material/database:",
        ),
        st.Page(
            "app_pages/basket_option_inputs.py",
            title="Basket option quote store",
            icon=":material/price_check:",
        ),
        st.Page(
            "app_pages/basket_covariance.py",
            title="Basket covariance validation",
            icon=":material/query_stats:",
        ),
        st.Page(
            "app_pages/basket_risk_hedging.py",
            title="Basket risk hedging",
            icon=":material/balance:",
        ),
        st.Page(
            "app_pages/basket_smile_hedging.py",
            title="Basket smile hedging",
            icon=":material/show_chart:",
        ),
        st.Page(
            "app_pages/model_benchmark.py",
            title="Model benchmark",
            icon=":material/compare_arrows:",
        ),
    ]
)

with st.sidebar:
    st.caption("VolForge v1.11.0 · Offline-first derivatives research")
    st.caption("Research software only — not investment or trading advice.")

page.run()
