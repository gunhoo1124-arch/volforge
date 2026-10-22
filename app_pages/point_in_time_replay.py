"""Point-in-time option-chain replay workspace."""

from __future__ import annotations

from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st

from volforge.data import SyntheticReplayConfig, generate_synthetic_replay
from volforge.replay import ReplaySettings, replay_report_json, run_chronological_replay
from volforge.reporting import dataframe_to_csv


@st.cache_data(max_entries=6, show_spinner=False)
def run_replay_lab(chain: pd.DataFrame, settings: ReplaySettings):
    """Run the complete chronological gate behind a bounded cache."""
    return run_chronological_replay(chain, settings=settings)


STRATEGY_LABELS = {
    "unhedged": "Unhedged",
    "delta": "Delta",
    "delta_gamma": "Delta-gamma",
    "delta_vega": "Delta-vega",
    "delta_gamma_vega": "Delta-gamma-vega",
}

st.title("Point-in-time replay")
st.caption(
    "Calibrate each chain snapshot with only contemporaneous quotes, hold fixed listed contracts, "
    "and produce a tamper-evident self-financing P&L ledger."
)

with st.form("point_in_time_replay_controls", border=True):
    st.subheader("Replay source", icon=":material/database:")
    source = st.segmented_control(
        "Chain source",
        ["Synthetic fixture", "Upload CSV"],
        default="Synthetic fixture",
        key="replay_source",
    )
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "Multi-snapshot option-chain CSV",
            type=["csv"],
            help="Every row must include quote_timestamp, expiration, strike, option_type, "
            "bid, ask, underlying_spot, risk_free_rate, and dividend_yield.",
        )

    control_columns = st.columns(3)
    with control_columns[0]:
        surface_model = st.segmented_control(
            "Total-variance surface", ["ssvi", "svi"], default="ssvi"
        )
        strategy_label = st.selectbox(
            "Listed hedge policy", list(STRATEGY_LABELS.values()), index=4
        )
    with control_columns[1]:
        stock_cost_bps = st.number_input(
            "Stock cost (bps of notional)", min_value=0.0, value=1.0, step=0.5
        )
        option_cost_bps = st.number_input(
            "Option cost (bps of premium)", min_value=0.0, value=5.0, step=1.0
        )
    with control_columns[2]:
        max_option_position = st.number_input(
            "Option position cap", min_value=0.1, value=50.0, step=5.0
        )
        calibration_budget = st.selectbox("Daily calibration budget", [60, 120, 240], index=0)
    submitted = st.form_submit_button(
        "Run chronological replay",
        type="primary",
        icon=":material/play_arrow:",
        key="replay_run",
    )

results_slot = st.container()
if not submitted:
    results_slot.info(
        "Choose a deterministic fixture or upload a multi-date chain, then run the replay. "
        "The fixture contains four daily snapshots and fixed expirations.",
        icon=":material/info:",
    )
    st.stop()

if source == "Upload CSV":
    if uploaded is None:
        results_slot.error("Upload a CSV before running the replay.", icon=":material/error:")
        st.stop()
    try:
        chain = pd.read_csv(BytesIO(uploaded.getvalue()))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        results_slot.error(f"Could not parse the CSV: {exc}", icon=":material/error:")
        st.stop()
else:
    chain = generate_synthetic_replay(SyntheticReplayConfig())

strategy = next(name for name, label in STRATEGY_LABELS.items() if label == strategy_label)
settings = ReplaySettings(
    surface_model=str(surface_model),
    strategy=strategy,
    stock_cost_bps=stock_cost_bps,
    option_cost_bps=option_cost_bps,
    max_option_position=max_option_position,
    heston_max_function_evaluations=int(calibration_budget),
    cos_terms=64,
)
try:
    with results_slot.skeleton(height=320):
        report = run_replay_lab(chain, settings)
except (ArithmeticError, RuntimeError, ValueError) as exc:
    results_slot.error(f"Replay failed to run: {exc}", icon=":material/error:")
    st.stop()

ledger = report.ledger_frame()
with results_slot:
    with st.container(horizontal=True):
        st.metric(
            "Replay gate",
            "Passed" if report.downstream_eligible else "Blocked",
            border=True,
        )
        st.metric(
            "Snapshots",
            f"{len(report.snapshots)} / {report.input_snapshot_count}",
            border=True,
        )
        st.metric(
            "Executable P&L",
            f"{report.total_pnl:.4f}" if report.total_pnl is not None else "Unavailable",
            border=True,
        )
        st.metric("Spread cost", f"{report.total_spread_cost:.4f}", border=True)
        st.metric("Transaction cost", f"{report.total_transaction_cost:.4f}", border=True)

    if report.target_contract is not None and report.hedge_contracts is not None:
        target_multiplier = report.snapshots[0].target_multiplier if report.snapshots else None
        multiplier_label = f"{target_multiplier:g}" if target_multiplier is not None else "unknown"
        st.caption(
            f"Short {report.target_contract.option_type} K={report.target_contract.strike:g}; "
            f"listed call hedges K={report.hedge_contracts[0].strike:g} and "
            f"K={report.hedge_contracts[1].strike:g}; expiry "
            f"{report.target_contract.expiration}; target multiplier "
            f"{multiplier_label}."
        )

    chart_frame = ledger.dropna(subset=["cumulative_pnl"])
    if not chart_frame.empty:
        chart_frame = chart_frame.assign(
            timestamp=pd.to_datetime(chart_frame["timestamp"], utc=True)
        )
        pnl_chart = (
            alt.Chart(chart_frame)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T", title="Snapshot time"),
                y=alt.Y("cumulative_pnl:Q", title="Cumulative P&L"),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Snapshot"),
                    alt.Tooltip("action:N", title="Action"),
                    alt.Tooltip("cumulative_pnl:Q", title="P&L", format=".6f"),
                    alt.Tooltip("spread_cost:Q", title="Spread cost", format=".6f"),
                    alt.Tooltip("transaction_cost:Q", title="Cost", format=".6f"),
                ],
            )
            .properties(height=320)
        )
        zero = alt.Chart(pd.DataFrame({"zero": [0.0]})).mark_rule().encode(y="zero:Q")
        with st.container(border=True):
            st.subheader("Executable-fill P&L", icon=":material/timeline:")
            st.altair_chart(
                pnl_chart + zero,
                alt="Cumulative bid-ask-side execution P&L across chronological snapshots",
            )

    with st.container(border=True):
        st.subheader("Tamper-evident daily audit", icon=":material/fingerprint:")
        visible_columns = [
            "sequence",
            "timestamp",
            "action",
            "surface_downstream_eligible",
            "heston_calibration_eligible",
            "target_multiplier",
            "hedge_low_multiplier",
            "hedge_high_multiplier",
            "spot",
            "target_mark",
            "target_fill",
            "stock_position",
            "hedge_low_position",
            "hedge_high_position",
            "spread_cost",
            "transaction_cost",
            "cumulative_pnl",
            "accounting_gap",
            "snapshot_hash",
            "audit_hash",
            "target_deliverable_hash",
            "failure_reasons",
        ]
        st.dataframe(
            ledger.loc[:, [column for column in visible_columns if column in ledger]],
            hide_index=True,
            key="replay_audit_ledger",
            alt="Chronological replay calibration, positions, P&L, and hash-chain audit",
            column_config={
                "snapshot_hash": st.column_config.TextColumn("Snapshot SHA-256"),
                "audit_hash": st.column_config.TextColumn("Audit SHA-256"),
                "accounting_gap": st.column_config.NumberColumn(format="%.3e"),
            },
        )
        st.caption(f"Final audit SHA-256: `{report.final_audit_hash or 'unavailable'}`")

    if report.downstream_eligible:
        st.success(
            "Every snapshot passed quote validation, static-arbitrage checks, Heston calibration, "
            "listed-quote availability, deliverable validation, multiplier-aware executable-side "
            "accounting, reconciliation, and audit-chain verification.",
            icon=":material/verified:",
        )
    else:
        st.error("Replay blocked: " + "; ".join(report.failure_reasons))

    with st.container(horizontal=True):
        st.download_button(
            "Download replay audit",
            data=replay_report_json(report),
            file_name="volforge-point-in-time-replay.json",
            mime="application/json",
            icon=":material/download:",
        )
        st.download_button(
            "Download P&L ledger",
            data=dataframe_to_csv(ledger),
            file_name="volforge-point-in-time-ledger.csv",
            mime="text/csv",
            icon=":material/download:",
        )

with st.expander("Replay interpretation", icon=":material/info:"):
    st.markdown(
        "- Contracts are selected from the first snapshot only and never replaced using future "
        "availability. A missing later mark blocks the replay.\n"
        "- Each day independently fits an arbitrage-gated total-variance surface and Heston model "
        "using only quotes carrying that exact timestamp.\n"
        "- Heston Greeks size positions. Option buys execute at the ask, sells at the bid, and "
        "open positions are marked at midpoint; the ledger reports spread cost separately from "
        "the proportional transaction-cost overlay.\n"
        "- Snapshot and chained audit hashes are tamper-evident records, not digital signatures or "
        "external timestamp attestations."
    )
