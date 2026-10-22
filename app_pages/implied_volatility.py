"""Robust implied-volatility inversion workspace."""

import streamlit as st

from volforge.implied_vol import implied_volatility

st.title("Implied-volatility solver", icon=":material/query_stats:")
st.caption("Prices are checked against European no-arbitrage bounds before inversion.")

model_label = st.segmented_control(
    "Model",
    ["Black-Scholes-Merton", "Bachelier"],
    default="Black-Scholes-Merton",
    key="iv_model",
)
model = "bsm" if model_label == "Black-Scholes-Merton" else "bachelier"

with st.form("iv_inputs"):
    first = st.columns(4)
    option_type = first[0].segmented_control(
        "Option type", ["call", "put"], default="call", key="iv_option_type"
    )
    option_price = first[1].number_input(
        "Observed option price", min_value=0.0, value=8.0, step=0.1
    )
    spot = first[2].number_input("Spot", min_value=0.01, value=100.0, step=1.0, key="iv_spot")
    strike = first[3].number_input("Strike", min_value=0.01, value=100.0, step=1.0, key="iv_strike")
    second = st.columns(4)
    days = second[0].number_input(
        "Days to expiry", min_value=0.0, value=90.0, step=1.0, key="iv_days"
    )
    rate = second[1].number_input(
        "Risk-free rate", value=0.03, step=0.005, format="%.4f", key="iv_rate"
    )
    dividend = second[2].number_input(
        "Dividend yield", value=0.01, step=0.005, format="%.4f", key="iv_dividend"
    )
    method = second[3].selectbox("Solver", ["brent", "newton"], key="iv_method")
    st.form_submit_button("Solve", type="primary", icon=":material/play_arrow:")

result = implied_volatility(
    option_price,
    option_type,
    spot,
    strike,
    days / 365.0,
    rate,
    dividend,
    model=model,
    method=method,
)

if result.converged and result.volatility is not None:
    label = "Implied lognormal volatility" if model == "bsm" else "Implied normal volatility"
    display = (
        f"{result.volatility:.6%}" if model == "bsm" else f"{result.volatility:,.6f} price/√year"
    )
    st.success(
        "The solver converged and repricing passed the residual check.",
        icon=":material/check_circle:",
    )
    with st.container(horizontal=True):
        st.metric(label, display, border=True)
        st.metric("Iterations", str(result.iterations), border=True)
        st.metric("Absolute residual", f"{abs(result.residual or 0.0):.3e}", border=True)
        st.metric("Method used", result.method_used, border=True)
else:
    st.error(
        f"No valid implied volatility: {result.failure_reason or 'unknown failure'}",
        icon=":material/error:",
    )
    st.table(
        {
            "Method": result.method_used,
            "Iterations": result.iterations,
            "Residual": "n/a" if result.residual is None else f"{result.residual:.6g}",
        },
        border="horizontal",
        width="content",
    )

st.caption(
    "Brent is the reliable default. Newton mode is safeguarded and falls back to a bracketed solve "
    "when vega is too small or an iteration would leave the valid bracket."
)
