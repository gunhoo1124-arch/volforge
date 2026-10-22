"""Interactive European option pricing workspace."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from volforge.models import bachelier, black_scholes

st.title("Pricing calculator", icon=":material/calculate:")
st.caption(
    "Compare Black-Scholes-Merton lognormal pricing with Bachelier normal pricing. "
    "Rates and dividend yields are continuously compounded."
)

model = st.segmented_control(
    "Model",
    options=["Black-Scholes-Merton", "Bachelier"],
    default="Black-Scholes-Merton",
    key="pricing_model",
)
model_key = "bsm" if model == "Black-Scholes-Merton" else "bachelier"

with st.form("pricing_inputs"):
    first = st.columns(3)
    option_type = first[0].segmented_control(
        "Option type", ["call", "put"], default="call", key="pricing_option_type"
    )
    spot = first[1].number_input("Spot", min_value=0.01, value=100.0, step=1.0)
    strike = first[2].number_input("Strike", min_value=0.01, value=100.0, step=1.0)
    second = st.columns(4)
    days = second[0].number_input("Days to expiry", min_value=0.0, value=90.0, step=1.0)
    rate = second[1].number_input("Risk-free rate", value=0.03, step=0.005, format="%.4f")
    dividend = second[2].number_input("Dividend yield", value=0.01, step=0.005, format="%.4f")
    volatility_label = (
        "Lognormal volatility" if model_key == "bsm" else "Normal volatility (price/√year)"
    )
    volatility_default = 0.20 if model_key == "bsm" else 20.0
    volatility = second[3].number_input(
        volatility_label,
        min_value=0.0,
        value=volatility_default,
        step=0.01 if model_key == "bsm" else 0.5,
        key=f"pricing_volatility_{model_key}",
    )
    st.form_submit_button("Recalculate", type="primary", icon=":material/play_arrow:")

maturity = days / 365.0
module = black_scholes if model_key == "bsm" else bachelier
try:
    result = module.price_and_greeks(
        option_type, spot, strike, maturity, volatility, rate, dividend
    )
    lower, upper = module.bounds(option_type, spot, strike, maturity, rate, dividend)
except ValueError as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

with st.container(horizontal=True):
    st.metric("Model price", f"{result.price:,.6f}", border=True)
    st.metric("Delta", f"{result.greeks.delta:,.6f}", border=True)
    st.metric("Gamma", f"{result.greeks.gamma:,.6f}", border=True)
    st.metric("Vega / 1 vol point", f"{result.greeks.vega / 100:,.6f}", border=True)
with st.container(horizontal=True):
    st.metric("Theta / year", f"{result.greeks.theta:,.6f}", border=True)
    st.metric("Theta / day", f"{result.greeks.theta / 365:,.6f}", border=True)
    st.metric("Rho / 1 rate point", f"{result.greeks.rho / 100:,.6f}", border=True)
    st.metric("No-arbitrage interval", f"[{lower:,.4f}, {upper:,.4f}]", border=True)

grid = np.linspace(max(0.01, spot * 0.4), spot * 1.6, 121)
values = [
    module.price(option_type, float(level), strike, maturity, volatility, rate, dividend)
    for level in grid
]
payoffs = (
    np.maximum(grid - strike, 0.0) if option_type == "call" else np.maximum(strike - grid, 0.0)
)
chart_data = pd.DataFrame(
    {
        "Underlying": np.concatenate([grid, grid]),
        "Value": np.concatenate([values, payoffs]),
        "Series": ["Model value"] * len(grid) + ["Expiry payoff"] * len(grid),
    }
)
chart = (
    alt.Chart(chart_data)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("Underlying:Q", title="Underlying spot"),
        y=alt.Y("Value:Q", title="Option value"),
        color=alt.Color("Series:N", title=None),
        strokeDash=alt.StrokeDash("Series:N", title=None),
        tooltip=[
            alt.Tooltip("Underlying:Q", format=".2f"),
            alt.Tooltip("Value:Q", format=".4f"),
            "Series:N",
        ],
    )
    .properties(height=360)
)
with st.container(border=True):
    st.subheader("Value profile", icon=":material/show_chart:")
    st.altair_chart(chart, alt="Current model value and expiry payoff across underlying prices")

with st.expander("Conventions and assumptions", icon=":material/info:"):
    st.markdown(
        "- European exercise with continuous rates and dividend yields.\n"
        "- Vega and rho are computed per absolute 1.00 move; cards divide them by 100.\n"
        "- Theta is calendar-time decay; the annual and daily conventions are shown.\n"
        "- Bachelier normal volatility is measured in underlying price units per square-root year, "
        "not a percentage. It permits normally distributed forward changes rather than lognormal levels."
    )
