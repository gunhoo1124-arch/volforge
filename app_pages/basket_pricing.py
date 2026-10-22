"""Pre-expiry pricing workspace for explicit adjusted deliverables."""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from volforge.basket import (
    BasketMonteCarloSettings,
    parse_basket_market,
    price_basket_option,
)
from volforge.deliverables import parse_deliverable_terms

PHYSICAL_DELIVERABLE = """[
  {"asset_id": "ACME", "asset_type": "security", "quantity": 50, "currency": "USD"},
  {"asset_id": "SPIN", "asset_type": "security", "quantity": 10, "currency": "USD"},
  {"asset_id": "USD-CIL", "asset_type": "cash_in_lieu", "quantity": 20, "currency": "USD"}
]"""
PHYSICAL_MARKET = """{
  "assets": [
    {"asset_id": "ACME", "spot": 120, "volatility": 0.25, "dividend_yield": 0.01},
    {"asset_id": "SPIN", "spot": 20, "volatility": 0.35, "dividend_yield": 0.00}
  ],
  "correlation": [[1.00, 0.35], [0.35, 1.00]]
}"""
CASH_DELIVERABLE = """[
  {"asset_id": "SPX-SET", "asset_type": "cash_settlement", "quantity": 100, "currency": "USD"}
]"""
CASH_MARKET = """{
  "assets": [
    {"asset_id": "SPX", "spot": 605, "volatility": 0.20, "dividend_yield": 0.015}
  ],
  "correlation": [[1.00]]
}"""


@st.cache_data(max_entries=12, show_spinner=False)
def _run_pricing(
    deliverable_json: str,
    market_json: str,
    settlement_method: str,
    option_type: str,
    strike: float,
    contract_multiplier: float,
    currency: str,
    underlying_id: str,
    maturity: float,
    rate: float,
    paths: int,
    seed: int,
    antithetic: bool,
    control_variate: bool,
):
    terms = parse_deliverable_terms(
        deliverable_json,
        settlement_method=settlement_method,
        contract_multiplier=contract_multiplier,
        currency=currency,
        underlying_id=underlying_id,
    )
    market = parse_basket_market(market_json)
    result = price_basket_option(
        terms,
        market,
        option_type=option_type,
        strike=strike,
        maturity=maturity,
        rate=rate,
        settings=BasketMonteCarloSettings(
            paths=paths,
            seed=seed,
            antithetic=antithetic,
            control_variate=control_variate,
        ),
    )
    return terms, market, result


st.title("Basket pricing")
st.caption(
    "Price European options on explicit adjusted deliverables under constant correlated GBM. "
    "Inputs are never inferred, and the result remains separate from Heston replay eligibility."
)

method_label = st.segmented_control(
    "Settlement method", ["Physical", "Cash"], default="Physical", key="basket_method"
)
settlement_method = str(method_label).lower()

with st.form("basket_pricing_controls", border=True):
    st.subheader("Contract and market", icon=":material/hub:")
    option_label = st.segmented_control(
        "Option type", ["Call", "Put"], default="Call", key="basket_option_type"
    )
    contract_columns = st.columns(4)
    with contract_columns[0]:
        strike = st.number_input(
            "Strike",
            min_value=0.0,
            value=60.0 if settlement_method == "physical" else 600.0,
            step=1.0,
            key=f"basket_strike_{settlement_method}",
        )
    with contract_columns[1]:
        multiplier = st.number_input(
            "Contract multiplier",
            min_value=0.000001,
            value=100.0,
            step=1.0,
            key="basket_multiplier",
        )
    with contract_columns[2]:
        maturity = st.number_input("Maturity (years)", min_value=0.0, value=0.5, step=0.05)
    with contract_columns[3]:
        rate = st.number_input("Risk-free rate", value=0.03, step=0.005, format="%.4f")

    identity_columns = st.columns(2)
    with identity_columns[0]:
        currency = st.text_input("Settlement currency", value="USD", max_chars=3)
    with identity_columns[1]:
        underlying_id = st.text_input(
            "Underlying ID",
            value="ACME" if settlement_method == "physical" else "SPX",
            help="Required risk factor for cash settlement; retained as contract identity otherwise.",
        )

    input_columns = st.columns(2)
    with input_columns[0]:
        deliverable_json = st.text_area(
            "Deliverable JSON",
            value=PHYSICAL_DELIVERABLE if settlement_method == "physical" else CASH_DELIVERABLE,
            height=220,
            key=f"basket_deliverable_{settlement_method}",
        )
    with input_columns[1]:
        market_json = st.text_area(
            "Basket market JSON",
            value=PHYSICAL_MARKET if settlement_method == "physical" else CASH_MARKET,
            height=220,
            key=f"basket_market_{settlement_method}",
            help="Asset order and correlation order must agree. Correlation must be PSD.",
        )

    sampling_columns = st.columns(4)
    with sampling_columns[0]:
        paths = st.selectbox("Monte Carlo paths", [20_000, 50_000, 100_000], index=0)
    with sampling_columns[1]:
        seed = st.number_input("Random seed", min_value=0, value=19, step=1)
    with sampling_columns[2]:
        antithetic = st.checkbox("Antithetic sampling", value=True)
    with sampling_columns[3]:
        control_variate = st.checkbox("Package control variate", value=True)

    submitted = st.form_submit_button(
        "Run basket pricing",
        type="primary",
        icon=":material/calculate:",
        key="basket_run",
    )

results_slot = st.container()
if not submitted:
    results_slot.info(
        "Review the explicit deliverable, market inputs, and correlation before running the model.",
        icon=":material/info:",
    )
    st.stop()

with results_slot:
    with st.spinner("Simulating correlated terminal package values...", show_time=True):
        try:
            terms, market, result = _run_pricing(
                deliverable_json,
                market_json,
                settlement_method,
                str(option_label).lower(),
                strike,
                multiplier,
                currency,
                underlying_id,
                maturity,
                rate,
                int(paths),
                int(seed),
                antithetic,
                control_variate,
            )
        except ValueError as exc:
            st.error(f"Basket pricing blocked: {exc}", icon=":material/error:")
            st.stop()

    with st.container(horizontal=True):
        st.metric("Monte Carlo price", f"{result.price:,.4f} {result.currency}", border=True)
        st.metric("Standard error", f"{result.standard_error:,.4f}", border=True)
        st.metric(
            f"{result.confidence_level:.0%} interval",
            f"[{result.confidence_interval_low:,.4f}, {result.confidence_interval_high:,.4f}]",
            border=True,
        )
        st.metric("Bounds gate", "Passed" if result.bounds_passed else "Failed", border=True)

    if result.bounds_passed:
        st.success(
            "The estimate passed the discounted lower and upper price bounds.",
            icon=":material/verified:",
        )
    else:
        st.error(
            "The estimate failed its price bounds and must not be used downstream.",
            icon=":material/error:",
        )

    diagnostic_columns = st.columns(2)
    with diagnostic_columns[0], st.container(border=True):
        st.subheader("Sampling diagnostics", icon=":material/query_stats:")
        st.metric("Crude estimate", f"{result.crude_price:,.4f} {result.currency}")
        st.metric("Crude standard error", f"{result.crude_standard_error:,.4f}")
        st.metric("Variance reduction", f"{result.variance_reduction_ratio:,.2f}x")
        st.caption(
            f"Seed {result.seed} · {result.paths:,} paths · "
            f"{result.independent_samples:,} independent samples · "
            f"control beta {result.control_beta:,.6f}"
        )
    with diagnostic_columns[1], st.container(border=True):
        st.subheader("Analytical cross-check", icon=":material/functions:")
        st.metric(
            "Moment-matched approximation",
            f"{result.moment_matched_price:,.4f} {result.currency}",
        )
        st.metric("Monte Carlo difference", f"{result.moment_match_difference:,.4f}")
        st.caption(
            f"No-arbitrage range: {result.lower_bound:,.4f} to "
            f"{result.upper_bound:,.4f} {result.currency}. The moment match is an "
            "approximation, not a second exact engine."
        )

    delta_frame = pd.DataFrame([asdict(delta) for delta in result.deltas])
    with st.container(border=True):
        st.subheader("Component deltas", icon=":material/speed:")
        if delta_frame.empty:
            st.info("This fixed-cash package has no simulated component delta.")
        else:
            st.dataframe(
                delta_frame,
                hide_index=True,
                key="basket_deltas",
                alt="Pathwise spot deltas and sampling errors for basket risk factors",
                column_config={
                    "asset_id": "Risk factor",
                    "delta": st.column_config.NumberColumn("Delta", format="%,.6f"),
                    "standard_error": st.column_config.NumberColumn(
                        "Standard error", format="%,.6f"
                    ),
                },
            )

    st.warning(
        "This static correlated-GBM result is not admitted to the chronological Heston replay "
        "ledger. Historical component marks, correlations, and hedge instruments are not yet "
        "represented there.",
        icon=":material/warning:",
    )

    audit_payload = {
        **result.to_dict(),
        "canonical_deliverable_json": terms.canonical_json,
        "canonical_market_json": market.canonical_json,
    }
    st.download_button(
        "Download pricing audit",
        data=json.dumps(audit_payload, indent=2, sort_keys=True) + "\n",
        file_name="volforge-basket-pricing.json",
        mime="application/json",
        icon=":material/download:",
    )

with st.expander("Model boundary", icon=":material/info:"):
    st.markdown(
        "- The engine prices European payoffs on positive security quantities and fixed cash under "
        "constant correlated risk-neutral geometric Brownian motion.\n"
        "- Every spot, volatility, dividend yield, and correlation is explicit. The engine rejects "
        "missing, unused, non-finite, or non-PSD market inputs.\n"
        "- The confidence interval describes Monte Carlo sampling error only; it does not include "
        "parameter, model, liquidity, or execution uncertainty.\n"
        "- The pathwise deltas are local sensitivities within this model and use the same package "
        "control variate as the price when enabled."
    )
