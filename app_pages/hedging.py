"""Seeded Heston hedging and P&L-attribution workspace."""

from __future__ import annotations

from dataclasses import asdict

import altair as alt
import pandas as pd
import streamlit as st

from volforge.hedging import HedgeBacktestSettings, run_heston_hedge_backtest
from volforge.models.heston import COSSettings, HestonParameters


@st.cache_data(max_entries=12, show_spinner=False)
def run_hedging_lab(
    option_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    parameters: HestonParameters,
    settings: HedgeBacktestSettings,
    cos_settings: COSSettings,
):
    """Run the common-path strategy comparison behind a bounded cache."""
    return run_heston_hedge_backtest(
        option_type,
        spot,
        strike,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=settings,
        cos_settings=cos_settings,
    )


STRATEGY_LABELS = {
    "unhedged": "Unhedged",
    "delta": "Delta",
    "delta_gamma": "Delta-gamma",
    "delta_vega": "Delta-vega",
    "delta_gamma_vega": "Delta-gamma-vega",
}

st.title("Hedging laboratory")
st.caption(
    "Compare self-financing Heston hedges on common seeded paths, including costs, position "
    "limits, conditioning diagnostics, and exactly reconciled terminal P&L attribution."
)

with st.form("hedging_controls"):
    st.subheader("Contract and model", icon=":material/tune:")
    contract_columns = st.columns(3)
    with contract_columns[0]:
        option_type = st.segmented_control(
            "Target option type", ["call", "put"], default="call", key="hedge_option_type"
        )
        spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
        strike = st.number_input("Target strike", min_value=0.01, value=100.0, step=1.0)
        maturity = st.number_input(
            "Maturity (ACT/365F years)", min_value=0.01, value=1.0, step=0.25
        )
    with contract_columns[1]:
        rate = st.number_input("Continuous rate", value=0.03, step=0.005, format="%.4f")
        dividend_yield = st.number_input(
            "Continuous dividend yield", value=0.0, step=0.005, format="%.4f"
        )
        kappa = st.number_input("Mean reversion (kappa)", min_value=0.001, value=1.5)
        theta = st.number_input("Long-run variance (theta)", min_value=0.0001, value=0.04)
    with contract_columns[2]:
        vol_of_vol = st.number_input("Volatility of variance", min_value=0.001, value=0.4)
        rho = st.number_input(
            "Spot/variance correlation", min_value=-0.999, max_value=0.999, value=-0.7
        )
        initial_variance = st.number_input("Initial variance", min_value=0.0001, value=0.04)
        seed = st.number_input("Simulation seed", min_value=0, value=7, step=1)

    st.subheader("Hedge design and costs", icon=":material/balance:")
    hedge_columns = st.columns(3)
    with hedge_columns[0]:
        paths = st.selectbox("Heston paths", [10, 20, 50, 100, 250], index=1)
        rebalancing_steps = st.selectbox("Rebalancing dates", [4, 8, 12, 26, 52], index=1)
        max_option_position = st.number_input(
            "Option position cap", min_value=0.1, value=50.0, step=5.0
        )
    with hedge_columns[1]:
        stock_cost_bps = st.number_input(
            "Stock cost (bps of notional)", min_value=0.0, value=1.0, step=0.5
        )
        option_cost_bps = st.number_input(
            "Option cost (bps of premium)", min_value=0.0, value=5.0, step=1.0
        )
        condition_limit = st.number_input(
            "Condition-number limit", min_value=10.0, value=1_000_000.0, step=10_000.0
        )
    with hedge_columns[2]:
        hedge_strike_low = st.number_input(
            "Low call strike / target strike",
            min_value=0.1,
            max_value=0.99,
            value=0.9,
            step=0.05,
        )
        hedge_strike_high = st.number_input(
            "High call strike / target strike",
            min_value=1.01,
            value=1.1,
            step=0.05,
        )
        cos_terms = st.selectbox("COS terms", [64, 128, 256, 512], index=1)
        cos_width = st.number_input(
            "COS truncation width", min_value=5.0, max_value=30.0, value=16.0, step=2.0
        )
    st.form_submit_button(
        "Run hedging laboratory",
        type="primary",
        icon=":material/play_arrow:",
        key="hedge_run_lab",
    )

parameters = HestonParameters(kappa, theta, vol_of_vol, rho, initial_variance)
settings = HedgeBacktestSettings(
    paths=int(paths),
    rebalancing_steps=int(rebalancing_steps),
    seed=int(seed),
    stock_cost_bps=stock_cost_bps,
    option_cost_bps=option_cost_bps,
    hedge_strike_low=hedge_strike_low,
    hedge_strike_high=hedge_strike_high,
    max_option_position=max_option_position,
    condition_limit=condition_limit,
)
cos_settings = COSSettings(int(cos_terms), cos_width)

results_slot = st.container()
try:
    with results_slot.skeleton(height=280):
        result = run_hedging_lab(
            str(option_type),
            spot,
            strike,
            maturity,
            rate,
            dividend_yield,
            parameters,
            settings,
            cos_settings,
        )
except (ArithmeticError, RuntimeError, ValueError) as exc:
    results_slot.error(f"Hedging backtest failed: {exc}", icon=":material/error:")
    st.stop()

strategy_map = {item.strategy: item for item in result.strategies}
comparison = pd.DataFrame(
    [
        {
            "Strategy": STRATEGY_LABELS[item.strategy],
            "Mean P&L": item.mean_pnl,
            "P&L standard deviation": item.pnl_standard_deviation,
            "Standard error": item.pnl_standard_error,
            "5% quantile": item.pnl_q05,
            "Median": item.pnl_median,
            "95% quantile": item.pnl_q95,
            "Mean transaction cost": item.mean_transaction_cost,
            "Mean turnover": item.mean_turnover,
            "Clipped rebalances": item.clipped_rebalances,
            "Ill-conditioned rebalances": item.ill_conditioned_rebalances,
        }
        for item in result.strategies
    ]
)
pnl_frame = pd.DataFrame(
    [
        {"Strategy": STRATEGY_LABELS[item.strategy], "Terminal P&L": pnl}
        for item in result.strategies
        for pnl in item.pnl
    ]
)
attribution_frame = pd.DataFrame(
    [
        {
            "Strategy": STRATEGY_LABELS[item.strategy],
            "Component": component.replace("_", " ").title(),
            "Mean contribution": value,
        }
        for item in result.strategies
        for component, value in asdict(item.mean_attribution).items()
        if component != "total"
    ]
)

with results_slot:
    unhedged = strategy_map["unhedged"]
    delta = strategy_map["delta"]
    full = strategy_map["delta_gamma_vega"]
    with st.container(horizontal=True):
        st.metric("Initial target premium", f"{result.initial_option_price:,.4f}", border=True)
        st.metric("Unhedged P&L volatility", f"{unhedged.pnl_standard_deviation:,.4f}", border=True)
        st.metric(
            "Delta P&L volatility",
            f"{delta.pnl_standard_deviation:,.4f}",
            delta=f"{delta.pnl_standard_deviation - unhedged.pnl_standard_deviation:+.4f}",
            delta_color="inverse",
            border=True,
        )
        st.metric(
            "Delta-gamma-vega volatility",
            f"{full.pnl_standard_deviation:,.4f}",
            delta=f"{full.pnl_standard_deviation - unhedged.pnl_standard_deviation:+.4f}",
            delta_color="inverse",
            border=True,
        )

    with st.container(border=True):
        st.subheader("Strategy comparison", icon=":material/compare_arrows:")
        st.dataframe(
            comparison,
            hide_index=True,
            key="hedge_strategy_comparison",
            alt="Terminal profit and loss, costs, and diagnostics by hedging strategy",
        )

    distribution_chart = (
        alt.Chart(pnl_frame)
        .mark_boxplot(extent="min-max", size=26)
        .encode(
            x=alt.X("Strategy:N", sort=list(STRATEGY_LABELS.values()), title=None),
            y=alt.Y("Terminal P&L:Q", title="Terminal P&L per short target option"),
            color=alt.Color("Strategy:N", legend=None),
            tooltip=["Strategy:N", alt.Tooltip("Terminal P&L:Q", format=".4f")],
        )
        .properties(height=340)
    )
    with st.container(border=True):
        st.subheader("Pathwise terminal P&L", icon=":material/candlestick_chart:")
        st.altair_chart(
            distribution_chart,
            alt="Box plots comparing pathwise terminal profit and loss across hedge strategies",
        )

    attribution_chart = (
        alt.Chart(attribution_frame)
        .mark_bar()
        .encode(
            x=alt.X("Strategy:N", sort=list(STRATEGY_LABELS.values()), title=None),
            y=alt.Y("sum(Mean contribution):Q", title="Mean P&L contribution"),
            color=alt.Color("Component:N", title="Component"),
            tooltip=[
                "Strategy:N",
                "Component:N",
                alt.Tooltip("Mean contribution:Q", format=".5f"),
            ],
        )
        .properties(height=340)
    )
    with st.container(border=True):
        st.subheader("Mean P&L attribution", icon=":material/account_tree:")
        st.altair_chart(
            attribution_chart,
            alt="Stacked mean delta, gamma, vega, theta, financing, cost, and residual P&L",
        )

    maximum_gap = max(item.max_abs_attribution_gap for item in result.strategies)
    if maximum_gap <= 1e-8:
        st.success(
            f"Every path reconciles to terminal P&L; maximum absolute gap {maximum_gap:.2e}.",
            icon=":material/verified:",
        )
    else:
        st.error(
            f"P&L attribution failed its reconciliation tolerance: {maximum_gap:.2e}.",
            icon=":material/error:",
        )
    clipped = sum(item.clipped_rebalances for item in result.strategies)
    ill_conditioned = sum(item.ill_conditioned_rebalances for item in result.strategies)
    if clipped or ill_conditioned:
        st.warning(
            f"Diagnostics: {clipped:,} option positions were clipped and {ill_conditioned:,} "
            "hedge solves used a stability fallback. Treat the affected strategies cautiously.",
            icon=":material/warning:",
        )

    with st.expander("Run diagnostics", icon=":material/diagnosis:"):
        st.json(
            {
                "Heston parameters": asdict(parameters),
                "COS settings": asdict(cos_settings),
                "Backtest": asdict(result),
            }
        )

with st.expander("Method and conventions", icon=":material/info:"):
    st.markdown(
        "- The experiment starts short one European target option. Delta hedges use stock; "
        "gamma and vega hedges use same-expiry calls at the two configured strikes.\n"
        "- Rebalancing is self-financing. Cash accrues at the continuous rate, stock receives "
        "a discrete continuous-yield credit, and costs apply to traded stock notional and option "
        "premium. Stock is liquidated at expiry.\n"
        "- Hedging vega is sensitivity to current `sqrt(v_t)` with long-run variance fixed. "
        "Theta is calendar decay per year. These differ deliberately from parallel surface-risk "
        "vega on the Heston Greeks page.\n"
        "- Delta, gamma, vega, and theta are local Taylor terms. Residual absorbs cross-Greeks, "
        "jumps between hedge dates, finite-difference error, and higher-order effects. Financing "
        "includes cash interest and stock dividends.\n"
        "- This is a risk-neutral synthetic experiment, not a historical trading backtest. "
        "Sampling error, model error, liquidity, discrete contracts, and market impact remain."
    )
