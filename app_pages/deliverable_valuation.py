"""Adjusted-deliverable settlement valuation workspace."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from volforge.deliverables import parse_deliverable_terms, value_option_settlement

PHYSICAL_EXAMPLE = """[
  {"asset_id": "ACME", "asset_type": "security", "quantity": 50, "currency": "USD"},
  {"asset_id": "SPIN", "asset_type": "security", "quantity": 10, "currency": "USD"},
  {"asset_id": "USD-CASH", "asset_type": "cash", "quantity": 15, "currency": "USD"},
  {"asset_id": "USD-CIL", "asset_type": "cash_in_lieu", "quantity": 5, "currency": "USD"}
]"""
PHYSICAL_MARKS = '{"ACME": 120, "SPIN": 20}'
CASH_EXAMPLE = """[
  {"asset_id": "SPX-SET", "asset_type": "cash_settlement", "quantity": 100, "currency": "USD"}
]"""

st.title("Deliverable valuation")
st.caption(
    "Value an adjusted physical package or a cash-settled payoff from explicit contract terms "
    "and marks. No missing component price or currency conversion is inferred."
)

method_label = st.segmented_control(
    "Settlement method",
    ["Physical", "Cash"],
    default="Physical",
    key="deliverable_method",
)
settlement_method = str(method_label).lower()

with st.form("deliverable_valuation_controls", border=True):
    st.subheader("Contract terms", icon=":material/contract:")
    option_label = st.segmented_control(
        "Option type",
        ["Call", "Put"],
        default="Call",
        key="deliverable_option_type",
    )
    term_columns = st.columns(4)
    with term_columns[0]:
        strike = st.number_input(
            "Strike",
            min_value=0.0,
            value=60.0 if settlement_method == "physical" else 600.0,
            step=1.0,
            key=f"deliverable_strike_{settlement_method}",
        )
    with term_columns[1]:
        multiplier = st.number_input(
            "Contract multiplier",
            min_value=0.000001,
            value=100.0,
            step=1.0,
        )
    with term_columns[2]:
        currency = st.text_input("Settlement currency", value="USD", max_chars=3)
    with term_columns[3]:
        underlying_id = st.text_input(
            "Underlying ID", value="ACME" if settlement_method == "physical" else "SPX"
        )

    if settlement_method == "physical":
        deliverable_json = st.text_area(
            "Deliverable JSON",
            value=PHYSICAL_EXAMPLE,
            height=180,
            key="physical_deliverable_json",
        )
        security_marks = st.text_area(
            "Security marks JSON",
            value=PHYSICAL_MARKS,
            height=100,
            help="One non-negative unit mark is required for every security component.",
        )
        settlement_level = None
    else:
        deliverable_json = st.text_area(
            "Deliverable JSON",
            value=CASH_EXAMPLE,
            height=130,
            key="cash_deliverable_json",
        )
        security_marks = None
        settlement_level = st.number_input(
            "Official settlement level",
            min_value=0.0,
            value=605.0,
            step=1.0,
            help="Use the official value defined by the product's settlement procedure.",
        )

    submitted = st.form_submit_button(
        "Value settlement",
        type="primary",
        icon=":material/calculate:",
        key="deliverable_value",
    )

results_slot = st.container()
if not submitted:
    results_slot.info(
        "Review the example terms, supply every required component mark, and value the settlement.",
        icon=":material/info:",
    )
    st.stop()

try:
    terms = parse_deliverable_terms(
        deliverable_json,
        settlement_method=settlement_method,
        contract_multiplier=multiplier,
        currency=currency,
        underlying_id=underlying_id,
    )
    valuation = value_option_settlement(
        terms,
        option_type=str(option_label).lower(),
        strike=strike,
        security_marks=security_marks,
        settlement_level=settlement_level,
    )
except ValueError as exc:
    results_slot.error(f"Settlement valuation blocked: {exc}", icon=":material/error:")
    st.stop()

with results_slot:
    with st.container(horizontal=True):
        st.metric(
            "Classification",
            valuation.classification.replace("_", " ").title(),
            border=True,
        )
        st.metric(
            "Deliverable value",
            f"{valuation.delivered_value:,.2f} {valuation.currency}",
            border=True,
        )
        st.metric(
            "Exercise price",
            f"{valuation.aggregate_exercise_price:,.2f} {valuation.currency}",
            border=True,
        )
        st.metric(
            "Net payoff",
            f"{valuation.net_payoff:,.2f} {valuation.currency}",
            border=True,
        )

    component_frame = pd.DataFrame(valuation.to_dict()["components"])
    with st.container(border=True):
        st.subheader("Component accounting", icon=":material/receipt_long:")
        st.dataframe(
            component_frame,
            hide_index=True,
            key="deliverable_components",
            alt="Unit marks and gross values for each settlement component",
            column_config={
                "quantity": st.column_config.NumberColumn("Quantity", format="%,.6f"),
                "unit_mark": st.column_config.NumberColumn("Unit mark", format="%,.6f"),
                "gross_value": st.column_config.NumberColumn("Gross value", format="%,.2f"),
            },
        )
        st.caption(f"Canonical deliverable SHA-256: `{terms.sha256}`")

    if terms.single_factor_pricing_compatible:
        st.success(
            "This is a standard single-security physical deliverable and is compatible with the "
            "current single-factor replay pricing gate.",
            icon=":material/verified:",
        )
    else:
        st.warning(
            "Settlement is valued, but this contract remains ineligible for the current "
            "single-factor pre-expiry Heston replay.",
            icon=":material/warning:",
        )

    audit_payload = {
        **valuation.to_dict(),
        "canonical_deliverable_json": terms.canonical_json,
        "required_security_marks": list(terms.required_security_marks),
        "single_factor_pricing_compatible": terms.single_factor_pricing_compatible,
    }
    st.download_button(
        "Download valuation audit",
        data=json.dumps(audit_payload, indent=2, sort_keys=True) + "\n",
        file_name="volforge-deliverable-valuation.json",
        mime="application/json",
        icon=":material/download:",
    )

with st.expander("Settlement interpretation", icon=":material/info:"):
    st.markdown(
        "- The contract multiplier scales the aggregate exercise price and cash-settlement "
        "difference; adjusted security quantities remain exactly as declared.\n"
        "- Fixed cash and resolved cash-in-lieu amounts are valued at face value in the settlement "
        "currency. Every security component requires an explicit unit mark.\n"
        "- Physical calls exchange the exercise cash for the package; physical puts exchange the "
        "package for the exercise cash. Cash-settled contracts pay only the positive strike-level "
        "difference.\n"
        "- This tool performs terminal settlement accounting. It does not price a multi-asset "
        "adjusted option before expiry or invent FX rates, component marks, or official settlement "
        "levels."
    )
