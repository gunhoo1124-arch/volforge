"""Dynamic basket smile-risk hedge workspace."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from volforge.basket_covariance import materialize_selected_covariance_replay
from volforge.basket_estimation import generate_synthetic_smile_basket_inputs
from volforge.basket_inputs import SYNTHETIC_BASKET_ID, BasketInputStore
from volforge.basket_replay import SYNTHETIC_BASKET_DELIVERABLE, SYNTHETIC_BASKET_EXPIRY
from volforge.basket_risk_hedging import (
    BasketRiskHedgeSettings,
    basket_risk_hedge_report_json,
    run_basket_risk_hedge_replay,
)
from volforge.deliverables import parse_deliverable_terms
from volforge.reporting import dataframe_to_csv

SOURCE = "volforge-synthetic-statistical-v1"


@st.cache_data(max_entries=4, show_spinner=False)
def _run_workspace(
    slope_bump: float,
    curvature_bump: float,
    paths: int,
) -> dict[str, object]:
    records, schedule = generate_synthetic_smile_basket_inputs()
    with tempfile.TemporaryDirectory(prefix="volforge-smile-basket-") as temporary:
        store = BasketInputStore.create(Path(temporary) / "basket-inputs.sqlite")
        store.ingest_frame(
            records,
            source_file_name="smile-basket-inputs.csv",
            source=SOURCE,
            mode="initial",
            ingested_timestamp="2026-05-01T00:00:00Z",
        )
        replay_input = materialize_selected_covariance_replay(
            store,
            schedule,
            basket_id=SYNTHETIC_BASKET_ID,
            source=SOURCE,
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
        settings=BasketRiskHedgeSettings(
            paths=paths,
            include_smile_risks=True,
            smile_slope_bump=slope_bump,
            smile_curvature_bump=curvature_bump,
            require_covariance_selection=True,
        ),
    )
    ledger = result.ledger_frame()
    instruments = result.instrument_frame()
    smile_profile = instruments.loc[
        instruments["sequence"] == 0,
        [
            "instrument_id",
            "forward_log_moneyness",
            "parallel_vega",
            "smile_skew_sensitivity",
            "smile_curvature_sensitivity",
            "target_position",
        ],
    ].sort_values("forward_log_moneyness")
    return {
        "replay_input": replay_input,
        "ledger": ledger,
        "instruments": instruments,
        "smile_profile": smile_profile,
        "components": result.component_frame(),
        "result": result.to_dict(),
        "report": basket_risk_hedge_report_json(result),
    }


st.title("Basket smile hedging")
st.caption(
    "Extend the executable basket hedge to dynamic forward-moneyness skew and curvature scenarios "
    "alongside gamma, parallel vega, correlation strength, and residual component delta."
)

with st.form("basket_smile_form"):
    with st.container(horizontal=True):
        slope_bump = st.selectbox(
            "Skew coefficient bump",
            (0.005, 0.01, 0.02),
            index=1,
            format_func=lambda value: f"{value:.3f}",
        )
        curvature_bump = st.selectbox(
            "Curvature coefficient bump",
            (0.005, 0.01, 0.02),
            index=1,
            format_func=lambda value: f"{value:.3f}",
        )
        paths = st.selectbox("Monte Carlo paths", (2_000, 4_000, 8_000), index=0)
    submitted = st.form_submit_button(
        "Run five-factor smile hedge",
        type="primary",
        icon=":material/show_chart:",
        key="basket_smile_run",
    )

result_slot = st.container()
st.warning(
    "Skew and curvature are explicit local scenarios, not a calibrated basket volatility surface. "
    "For log-moneyness k, the common component-volatility shift is slope * k plus curvature * k^2. "
    "Passing the hedge gate does not establish smile-model adequacy or executable liquidity."
)

if submitted:
    try:
        with result_slot.skeleton():
            workspace = _run_workspace(slope_bump, curvature_bump, paths)
    except (ArithmeticError, OSError, RuntimeError, ValueError) as exc:
        result_slot.error(f"Smile-risk hedge blocked: {exc}")
    else:
        result = workspace["result"]
        ledger = workspace["ledger"]
        instruments = workspace["instruments"]
        smile_profile = workspace["smile_profile"]
        components = workspace["components"]
        assert isinstance(result, dict)
        assert isinstance(ledger, pd.DataFrame)
        assert isinstance(instruments, pd.DataFrame)
        assert isinstance(smile_profile, pd.DataFrame)
        assert isinstance(components, pd.DataFrame)
        condition_numbers = ledger["solver_condition_number"].dropna()
        worst_condition = (
            f"{float(condition_numbers.max()):,.2f}" if not condition_numbers.empty else "N/A"
        )
        total_pnl = result["total_pnl"]
        final_pnl = "N/A" if total_pnl is None else f"${float(total_pnl):,.2f}"
        with result_slot:
            with st.container(horizontal=True):
                st.metric(
                    "Replay gate",
                    "Passed" if result["downstream_eligible"] else "Blocked",
                    border=True,
                )
                st.metric(
                    "Smile-risk coverage",
                    "Bound" if result["smile_risk_coverage_bound"] else "Missing",
                    border=True,
                )
                st.metric(
                    "Worst condition",
                    worst_condition,
                    border=True,
                )
                st.metric(
                    "Final P&L",
                    final_pnl,
                    border=True,
                )
            if result["downstream_eligible"]:
                st.success(
                    "All five declared nonlinear risks had full-rank instrument coverage, passed "
                    "conditioning and position gates, and were neutralized within tolerance."
                )
            else:
                st.error("Smile hedge blocked: " + "; ".join(result["failure_reasons"]))

            st.subheader("Dynamic smile coordinates and sensitivities")
            st.line_chart(
                smile_profile,
                x="forward_log_moneyness",
                y=["smile_skew_sensitivity", "smile_curvature_sensitivity"],
                alt="Skew and curvature sensitivity across hedge-option forward log-moneyness",
            )
            st.dataframe(
                smile_profile,
                hide_index=True,
                key="basket_smile_profile",
                alt="Opening hedge instruments with smile coordinates, risks, and positions",
                column_config={
                    "forward_log_moneyness": st.column_config.NumberColumn(format="%.6f"),
                    "parallel_vega": st.column_config.NumberColumn(format="%.6f"),
                    "smile_skew_sensitivity": st.column_config.NumberColumn(format="%.6f"),
                    "smile_curvature_sensitivity": st.column_config.NumberColumn(format="%.6f"),
                    "target_position": st.column_config.NumberColumn(format="%.6f"),
                },
            )

            st.subheader("Rank-aware hedge solves")
            st.dataframe(
                ledger[
                    [
                        "formation_timestamp",
                        "action",
                        "solver_rank",
                        "solver_minimum_singular_value",
                        "solver_condition_number",
                        "normalized_solver_residual",
                        "residual_smile_skew_sensitivity",
                        "residual_smile_curvature_sensitivity",
                        "accounting_gap",
                    ]
                ],
                hide_index=True,
                key="basket_smile_solver",
                alt="Rank, singular value, condition, residual, and accounting diagnostics",
            )

            st.subheader("Executable hedge positions")
            st.dataframe(
                instruments[
                    [
                        "formation_timestamp",
                        "instrument_id",
                        "bid",
                        "ask",
                        "forward_log_moneyness",
                        "target_position",
                        "trade",
                    ]
                ],
                hide_index=True,
                key="basket_smile_instruments",
                alt="Five quoted basket-option positions and adverse-side trades",
            )

            with st.container(horizontal=True):
                st.download_button(
                    "Smile-selected replay input",
                    data=dataframe_to_csv(workspace["replay_input"]),
                    file_name="volforge-smile-replay-input.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                st.download_button(
                    "Smile hedge ledger",
                    data=dataframe_to_csv(ledger),
                    file_name="volforge-smile-hedge-ledger.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                st.download_button(
                    "Hash-chained audit",
                    data=workspace["report"],
                    file_name="volforge-smile-hedge-audit.json",
                    mime="application/json",
                    icon=":material/download:",
                )

st.caption(
    "The fixture is deterministic and offline. Production research must source independent "
    "smile-sensitive option quotes and justify the scenario definition, bumps, and position limits."
)
