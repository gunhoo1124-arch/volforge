"""Statistical basket estimation and multi-instrument risk hedge workspace."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from volforge.basket_estimation import (
    BasketEstimationSettings,
    generate_synthetic_statistical_basket_inputs,
    materialize_estimated_replay,
)
from volforge.basket_inputs import SYNTHETIC_BASKET_ID, BasketInputStore
from volforge.basket_replay import SYNTHETIC_BASKET_DELIVERABLE, SYNTHETIC_BASKET_EXPIRY
from volforge.basket_risk_hedging import (
    BasketRiskHedgeSettings,
    basket_risk_hedge_report_json,
    run_basket_risk_hedge_replay,
)
from volforge.deliverables import parse_deliverable_terms
from volforge.reporting import dataframe_to_csv


@st.cache_data(max_entries=4, show_spinner=False)
def _run_workspace(window_returns: int, shrinkage: float, paths: int) -> dict[str, object]:
    records, schedule = generate_synthetic_statistical_basket_inputs()
    with tempfile.TemporaryDirectory(prefix="volforge-statistical-basket-") as temporary:
        store = BasketInputStore.create(Path(temporary) / "basket-inputs.sqlite")
        store.ingest_frame(
            records,
            source_file_name="statistical-basket-inputs.csv",
            source="volforge-synthetic-statistical-v1",
            mode="initial",
            ingested_timestamp="2026-05-01T00:00:00Z",
        )
        replay_input = materialize_estimated_replay(
            store,
            schedule,
            basket_id=SYNTHETIC_BASKET_ID,
            source="volforge-synthetic-statistical-v1",
            settings=BasketEstimationSettings(
                window_returns=window_returns,
                minimum_returns=10,
                correlation_shrinkage=shrinkage,
            ),
        )
    terms = parse_deliverable_terms(
        SYNTHETIC_BASKET_DELIVERABLE,
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ACME",
    )
    result = run_basket_risk_hedge_replay(
        replay_input,
        terms,
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=BasketRiskHedgeSettings(paths=paths),
    )
    estimates = pd.DataFrame([json.loads(value) for value in replay_input["input_estimate_json"]])
    estimate_summary = pd.DataFrame(
        {
            "formation_timestamp": estimates["formation_timestamp"],
            "return_observation_count": estimates["return_observation_count"],
            "observation_start": estimates["observation_start"],
            "observation_end": estimates["observation_end"],
            "sample_minimum_eigenvalue": estimates["sample_minimum_eigenvalue"],
            "shrunk_minimum_eigenvalue": estimates["shrunk_minimum_eigenvalue"],
            "estimate_sha256": estimates["estimate_sha256"],
        }
    )
    return {
        "records": records,
        "schedule": schedule,
        "replay_input": replay_input,
        "estimate_summary": estimate_summary,
        "ledger": result.ledger_frame(),
        "instruments": result.instrument_frame(),
        "components": result.component_frame(),
        "result": result.to_dict(),
        "report": basket_risk_hedge_report_json(result),
    }


st.title("Basket risk hedging")
st.caption(
    "Estimate point-in-time basket volatility and correlation, then neutralize component delta "
    "plus aggregate gamma, parallel vega, and correlation sensitivity with executable quotes."
)

with st.form("basket_risk_hedge_form"):
    with st.container(horizontal=True):
        window_returns = st.selectbox(
            "Rolling return window",
            (15, 20, 30),
            index=1,
            help="Only synchronized returns received by each formation time are eligible.",
        )
        shrinkage = st.selectbox(
            "Correlation shrinkage",
            (0.10, 0.20, 0.35),
            index=1,
            format_func=lambda value: f"{value:.0%}",
        )
        paths = st.selectbox("Monte Carlo paths", (2_000, 4_000, 8_000), index=1)
    submitted = st.form_submit_button(
        "Run estimated risk hedge",
        type="primary",
        icon=":material/balance:",
        key="basket_risk_run",
    )

result_slot = st.container()
st.warning(
    "This is a controlled research hedge, not a claim of market completeness. The three option "
    "instruments span only the declared aggregate bump risks; basis, jump, liquidity, and model "
    "risk remain outside the neutralization system."
)

if submitted:
    try:
        with result_slot.skeleton():
            workspace = _run_workspace(window_returns, shrinkage, paths)
    except (ArithmeticError, OSError, RuntimeError, ValueError) as exc:
        result_slot.error(f"Advanced basket gate blocked: {exc}")
    else:
        result = workspace["result"]
        ledger = workspace["ledger"]
        instruments = workspace["instruments"]
        components = workspace["components"]
        estimates = workspace["estimate_summary"]
        assert isinstance(result, dict)
        assert isinstance(ledger, pd.DataFrame)
        assert isinstance(instruments, pd.DataFrame)
        assert isinstance(components, pd.DataFrame)
        assert isinstance(estimates, pd.DataFrame)
        with result_slot:
            with st.container(horizontal=True):
                st.metric(
                    "Replay gate",
                    "Passed" if result["downstream_eligible"] else "Blocked",
                    border=True,
                )
                st.metric(
                    "Estimated inputs",
                    "Bound" if result["estimate_provenance_bound"] else "Missing",
                    border=True,
                )
                st.metric(
                    "Max exposure residual",
                    f"{float(result['maximum_normalized_exposure_residual']):.2e}",
                    border=True,
                )
                st.metric(
                    "Final P&L",
                    f"${float(result['total_pnl']):,.2f}",
                    border=True,
                )
            if result["downstream_eligible"]:
                st.success(
                    "All four estimates passed no-lookahead and PSD checks; the nonlinear solver, "
                    "position limits, accounting identity, and audit chain also passed."
                )
            else:
                st.error("The advanced hedge was blocked: " + "; ".join(result["failure_reasons"]))

            st.subheader("Point-in-time estimates")
            st.dataframe(
                estimates,
                hide_index=True,
                key="basket_risk_estimates",
                alt="Formation-date basket estimates and provenance identities",
                column_config={
                    "sample_minimum_eigenvalue": st.column_config.NumberColumn(format="%.6f"),
                    "shrunk_minimum_eigenvalue": st.column_config.NumberColumn(format="%.6f"),
                },
            )
            st.subheader("Self-financing ledger")
            st.line_chart(
                ledger,
                x="formation_timestamp",
                y="portfolio_value",
                alt="Advanced basket hedge portfolio value through target settlement",
            )
            st.dataframe(
                ledger[
                    [
                        "formation_timestamp",
                        "action",
                        "solver_condition_number",
                        "normalized_solver_residual",
                        "spread_cost",
                        "transaction_cost",
                        "portfolio_value",
                        "accounting_gap",
                    ]
                ],
                hide_index=True,
                key="basket_risk_ledger",
                alt="Risk solver and self-financing accounting ledger",
            )
            st.subheader("Explicit option hedge positions")
            st.dataframe(
                instruments[
                    [
                        "formation_timestamp",
                        "instrument_id",
                        "bid",
                        "ask",
                        "target_position",
                        "trade",
                        "transaction_cost",
                    ]
                ],
                hide_index=True,
                key="basket_risk_instruments",
                alt="Quoted gamma, vega, and correlation hedge option trades",
            )
            st.subheader("Residual component delta hedges")
            st.dataframe(
                components[
                    [
                        "formation_timestamp",
                        "asset_id",
                        "net_option_delta_before_component",
                        "target_position",
                        "trade",
                    ]
                ],
                hide_index=True,
                key="basket_risk_components",
                alt="Component positions that neutralize residual option delta",
            )
            with st.container(horizontal=True):
                st.download_button(
                    "Estimated replay input",
                    data=dataframe_to_csv(workspace["replay_input"]),
                    file_name="volforge-estimated-basket-replay.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                st.download_button(
                    "Advanced hedge ledger",
                    data=dataframe_to_csv(ledger),
                    file_name="volforge-basket-risk-ledger.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                st.download_button(
                    "Hash-chained audit",
                    data=workspace["report"],
                    file_name="volforge-basket-risk-audit.json",
                    mime="application/json",
                    icon=":material/download:",
                )

st.caption(
    "The fixture is deterministic and offline. Production schedules must supply independently "
    "controlled target and hedge-instrument quotes; model-derived demo quotes are not executable "
    "market evidence."
)
