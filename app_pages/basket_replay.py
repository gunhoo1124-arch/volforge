"""Point-in-time adjusted-option hedge replay workspace."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from volforge.basket_replay import (
    SYNTHETIC_BASKET_DELIVERABLE,
    SYNTHETIC_BASKET_EXPIRY,
    BasketReplaySettings,
    basket_replay_report_json,
    generate_synthetic_basket_replay,
    run_basket_hedge_replay,
    verify_basket_replay_hash_chain,
)
from volforge.deliverables import parse_deliverable_terms
from volforge.reporting import dataframe_to_csv


@st.cache_data(max_entries=4, show_spinner=False)
def _synthetic_csv() -> bytes:
    return dataframe_to_csv(generate_synthetic_basket_replay())


@st.cache_data(max_entries=8, show_spinner=False)
def _run_replay(
    csv_payload: bytes,
    deliverable_json: str,
    settlement_method: str,
    option_type: str,
    strike: float,
    contract_multiplier: float,
    currency: str,
    underlying_id: str,
    expiry_timestamp: str,
    paths: int,
    seed: int,
    component_cost_bps: float,
    option_cost_bps: float,
    maximum_market_age_seconds: float,
):
    terms = parse_deliverable_terms(
        deliverable_json,
        settlement_method=settlement_method,
        contract_multiplier=contract_multiplier,
        currency=currency,
        underlying_id=underlying_id,
    )
    result = run_basket_hedge_replay(
        pd.read_csv(io.BytesIO(csv_payload)),
        terms,
        option_type=option_type,
        strike=strike,
        expiry_timestamp=expiry_timestamp,
        settings=BasketReplaySettings(
            paths=paths,
            seed=seed,
            component_cost_bps=component_cost_bps,
            option_cost_bps=option_cost_bps,
            maximum_market_age_seconds=maximum_market_age_seconds,
        ),
    )
    return terms, result


st.title("Basket hedge replay")
st.caption(
    "Replay one short adjusted option through expiry using only market packages received by each "
    "formation time, explicit component bid/ask execution, and model component deltas."
)

source_mode = st.segmented_control(
    "Snapshot source",
    ["Synthetic fixture", "Upload CSV"],
    default="Synthetic fixture",
    key="basket_replay_source",
)
uploaded = None
if source_mode == "Upload CSV":
    uploaded = st.file_uploader(
        "Basket replay CSV",
        type=["csv"],
        help="Use the documented formation, event, receipt, market, and execution schema.",
    )

with st.form("basket_replay_controls", border=True):
    st.subheader("Replay contract", icon=":material/contract:")
    settlement_label = st.segmented_control(
        "Settlement method",
        ["Physical", "Cash"],
        default="Physical",
        key="basket_replay_settlement",
    )
    option_label = st.segmented_control(
        "Option type", ["Call", "Put"], default="Call", key="basket_replay_option_type"
    )
    contract_columns = st.columns(4)
    with contract_columns[0]:
        strike = st.number_input("Strike", min_value=0.0, value=60.0, step=1.0)
    with contract_columns[1]:
        multiplier = st.number_input(
            "Contract multiplier", min_value=0.000001, value=100.0, step=1.0
        )
    with contract_columns[2]:
        currency = st.text_input("Settlement currency", value="USD", max_chars=3)
    with contract_columns[3]:
        underlying_id = st.text_input("Underlying ID", value="ACME")

    expiry_timestamp = st.text_input(
        "Expiry timestamp",
        value=SYNTHETIC_BASKET_EXPIRY,
        help="Timezone-aware timestamp that must equal the final formation timestamp.",
    )
    deliverable_json = st.text_area(
        "Deliverable JSON",
        value=SYNTHETIC_BASKET_DELIVERABLE,
        height=150,
    )

    st.subheader("Sampling and execution", icon=":material/tune:")
    setting_columns = st.columns(5)
    with setting_columns[0]:
        paths = st.selectbox("Paths per snapshot", [4_000, 10_000, 20_000], index=0)
    with setting_columns[1]:
        seed = st.number_input("Base seed", min_value=0, value=31, step=1)
    with setting_columns[2]:
        component_cost_bps = st.number_input(
            "Component cost (bps)", min_value=0.0, value=1.0, step=0.5
        )
    with setting_columns[3]:
        option_cost_bps = st.number_input("Option cost (bps)", min_value=0.0, value=5.0, step=0.5)
    with setting_columns[4]:
        maximum_market_age = st.number_input(
            "Maximum market age (s)", min_value=0.0, value=900.0, step=60.0
        )

    submitted = st.form_submit_button(
        "Run basket replay",
        type="primary",
        icon=":material/route:",
        key="basket_replay_run",
    )

results_slot = st.container()
if not submitted:
    results_slot.info(
        "Use the deterministic four-snapshot fixture or upload a point-in-time basket CSV.",
        icon=":material/info:",
    )
    st.stop()

if source_mode == "Upload CSV" and uploaded is None:
    results_slot.error("Upload a basket replay CSV before running.", icon=":material/error:")
    st.stop()

csv_payload = uploaded.getvalue() if uploaded is not None else _synthetic_csv()
with results_slot:
    with st.spinner("Repricing and reconciling the chronological hedge ledger...", show_time=True):
        try:
            terms, result = _run_replay(
                csv_payload,
                deliverable_json,
                str(settlement_label).lower(),
                str(option_label).lower(),
                strike,
                multiplier,
                currency,
                underlying_id,
                expiry_timestamp,
                int(paths),
                int(seed),
                component_cost_bps,
                option_cost_bps,
                maximum_market_age,
            )
        except ValueError as exc:
            st.error(f"Basket replay blocked: {exc}", icon=":material/error:")
            st.stop()

    with st.container(horizontal=True):
        st.metric(
            "Replay gate",
            "Passed" if result.downstream_eligible else "Blocked",
            border=True,
        )
        st.metric(
            "Terminal P&L",
            f"{result.total_pnl:,.2f} {result.currency}"
            if result.total_pnl is not None
            else "Unavailable",
            border=True,
        )
        st.metric("Spread cost", f"{result.total_spread_cost:,.2f}", border=True)
        st.metric("Explicit cost", f"{result.total_transaction_cost:,.2f}", border=True)
        st.metric(
            "Maximum accounting gap",
            f"{result.maximum_absolute_accounting_gap:.2e}"
            if result.maximum_absolute_accounting_gap is not None
            else "Unavailable",
            border=True,
        )

    if result.downstream_eligible:
        st.success(
            "Every supplied snapshot passed timing, pricing, execution, settlement, accounting, "
            "and audit-chain checks.",
            icon=":material/verified:",
        )
    else:
        st.error(
            "Replay stopped at the first invalid snapshot: " + "; ".join(result.failure_reasons),
            icon=":material/block:",
        )

    ledger = result.ledger_frame()
    if not ledger.empty:
        chart_frame = ledger[["formation_timestamp", "portfolio_value"]].copy()
        chart_frame["formation_timestamp"] = pd.to_datetime(
            chart_frame["formation_timestamp"], utc=True
        )
        with st.container(border=True):
            st.subheader("Hedged portfolio value", icon=":material/show_chart:")
            st.line_chart(
                chart_frame,
                x="formation_timestamp",
                y="portfolio_value",
                x_label="Formation time",
                y_label=f"Value ({result.currency})",
                alt="Chronological marked value of the adjusted-option hedge portfolio",
            )

        display_columns = [
            "sequence",
            "formation_timestamp",
            "action",
            "option_mid",
            "option_model_price",
            "portfolio_value",
            "spread_cost",
            "transaction_cost",
            "accounting_gap",
        ]
        with st.container(border=True):
            st.subheader("Snapshot ledger", icon=":material/receipt_long:")
            st.dataframe(
                ledger[display_columns],
                hide_index=True,
                key="basket_replay_ledger",
                alt="Chronological basket hedge pricing and accounting ledger",
                column_config={
                    "option_mid": st.column_config.NumberColumn(
                        "Observed option mid", format="%,.4f"
                    ),
                    "option_model_price": st.column_config.NumberColumn(
                        "Model price", format="%,.4f"
                    ),
                    "portfolio_value": st.column_config.NumberColumn(
                        "Portfolio value", format="%,.4f"
                    ),
                    "spread_cost": st.column_config.NumberColumn("Spread cost", format="%,.4f"),
                    "transaction_cost": st.column_config.NumberColumn(
                        "Explicit cost", format="%,.4f"
                    ),
                    "accounting_gap": st.column_config.NumberColumn(
                        "Accounting gap", format="%.3e"
                    ),
                },
            )

        components = result.component_frame()
        with st.container(border=True):
            st.subheader("Component hedge trades", icon=":material/balance:")
            st.dataframe(
                components,
                hide_index=True,
                key="basket_replay_components",
                alt="Delta targets and executable trades for each basket component",
                column_config={
                    name: st.column_config.NumberColumn(format="%,.6f")
                    for name in (
                        "model_delta",
                        "delta_standard_error",
                        "prior_position",
                        "target_position",
                        "trade",
                    )
                },
            )

        with st.container(horizontal=True):
            st.download_button(
                "Download ledger CSV",
                data=dataframe_to_csv(ledger),
                file_name="volforge-basket-replay-ledger.csv",
                mime="text/csv",
                icon=":material/download:",
            )
            st.download_button(
                "Download replay audit",
                data=basket_replay_report_json(result),
                file_name="volforge-basket-replay-audit.json",
                mime="application/json",
                icon=":material/download:",
            )

        st.caption(
            f"Deliverable SHA-256: `{terms.sha256}` · final audit hash: "
            f"`{result.final_audit_hash}` · chain verified: "
            f"`{verify_basket_replay_hash_chain(result)}`"
        )

st.warning(
    "Source labels and record IDs preserve supplied provenance but do not authenticate a vendor. "
    "This adjusted-package ledger is separate from the SVI/SSVI-Heston replay and does not claim "
    "that user-supplied volatilities or correlations were historically observable.",
    icon=":material/warning:",
)

with st.expander("Replay conventions", icon=":material/info:"):
    st.markdown(
        "- One short option is sold at the first observed bid, held through expiry, and settled "
        "from the explicit deliverable and final component marks.\n"
        "- Component hedge buys execute at ask and sells at bid. Cash earns the prior snapshot's "
        "continuous rate; held components receive the prior declared continuous dividend yield.\n"
        "- Market event time must precede receipt time, which must not exceed formation time. "
        "Stale packages and changing risk-factor identities block immediately.\n"
        "- Passing proves this finite supplied ledger reconciles and its audit chain verifies. It "
        "does not establish model adequacy, source authenticity, or executable future performance."
    )
