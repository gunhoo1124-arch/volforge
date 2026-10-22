"""Rolling out-of-sample BSM-versus-Heston benchmark workspace."""

from __future__ import annotations

from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st

from volforge.benchmark import (
    COMPARISON_LOSSES,
    BenchmarkSettings,
    benchmark_report_json,
    run_chronological_benchmark,
)
from volforge.data import SyntheticReplayConfig, generate_synthetic_replay
from volforge.reporting import dataframe_to_csv
from volforge.research import build_benchmark_research_bundle


@st.cache_data(max_entries=6, show_spinner=False)
def run_benchmark_lab(chain: pd.DataFrame, settings: BenchmarkSettings):
    """Run the rolling benchmark behind a bounded cache."""
    return run_chronological_benchmark(chain, settings=settings)


st.title("Model benchmark")
st.caption(
    "Fit BSM and Heston only on trailing snapshots, price the untouched next chain, and compare "
    "dollar error with quoted bid/ask-band loss."
)

with st.form("model_benchmark_controls", border=True):
    st.subheader("Benchmark source", icon=":material/compare_arrows:")
    source = st.segmented_control(
        "Chain source",
        ["Synthetic fixture", "Upload CSV"],
        default="Synthetic fixture",
        key="benchmark_source",
    )
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "Multi-snapshot option-chain CSV",
            type=["csv"],
            help="Every row must include quote_timestamp, expiration, strike, option_type, "
            "bid, ask, underlying_spot, risk_free_rate, and dividend_yield.",
        )
    control_columns = st.columns(4)
    with control_columns[0]:
        surface_model = st.segmented_control("Training surface", ["ssvi", "svi"], default="ssvi")
    with control_columns[1]:
        training_window = st.number_input(
            "Trailing snapshots", min_value=1, max_value=60, value=2, step=1
        )
    with control_columns[2]:
        calibration_budget = st.selectbox("Per-snapshot Heston budget", [60, 120, 240], index=0)
    with control_columns[3]:
        comparison_loss = st.selectbox(
            "Primary comparison loss",
            COMPARISON_LOSSES,
            format_func=lambda value: str(value).replace("_", " ").capitalize(),
        )
    submitted = st.form_submit_button(
        "Run out-of-sample benchmark",
        type="primary",
        icon=":material/play_arrow:",
        key="benchmark_run",
    )

results_slot = st.container()
if not submitted:
    results_slot.info(
        "The deterministic fixture has four chain snapshots. With the default two-snapshot "
        "window, the final two snapshots are scored without same-day calibration.",
        icon=":material/info:",
    )
    st.stop()

if source == "Upload CSV":
    if uploaded is None:
        results_slot.error("Upload a CSV before running the benchmark.", icon=":material/error:")
        st.stop()
    try:
        chain = pd.read_csv(BytesIO(uploaded.getvalue()))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        results_slot.error(f"Could not parse the CSV: {exc}", icon=":material/error:")
        st.stop()
else:
    chain = generate_synthetic_replay(SyntheticReplayConfig())

settings = BenchmarkSettings(
    training_window=int(training_window),
    surface_model=str(surface_model),
    heston_max_function_evaluations=int(calibration_budget),
    cos_terms=64,
    comparison_loss=str(comparison_loss),
)
try:
    with results_slot.skeleton(height=320):
        report = run_benchmark_lab(chain, settings)
except (ArithmeticError, RuntimeError, ValueError) as exc:
    results_slot.error(f"Benchmark failed to run: {exc}", icon=":material/error:")
    st.stop()

predictions = report.prediction_frame()
summary = report.summary_frame()
timeline = report.timeline_frame()
panels = report.panel_frame()
inference = report.comparison_inference()

with results_slot:
    bsm_row = summary.loc[summary["model"] == "bsm"]
    heston_row = summary.loc[summary["model"] == "heston"]
    bsm_rmse = float(bsm_row["rmse"].iloc[0]) if not bsm_row.empty else None
    heston_rmse = float(heston_row["rmse"].iloc[0]) if not heston_row.empty else None
    improvement = (
        1.0 - heston_rmse / bsm_rmse
        if bsm_rmse is not None and heston_rmse is not None and bsm_rmse > 0.0
        else None
    )
    with st.container(horizontal=True):
        st.metric(
            "Benchmark gate",
            "Passed" if report.downstream_eligible else "Blocked",
            border=True,
        )
        st.metric("Scored snapshots", report.evaluation_snapshot_count, border=True)
        st.metric("BSM RMSE", f"{bsm_rmse:.4f}" if bsm_rmse is not None else "—", border=True)
        st.metric(
            "Heston RMSE",
            f"{heston_rmse:.4f}" if heston_rmse is not None else "—",
            border=True,
        )
        st.metric(
            "Heston RMSE reduction",
            f"{improvement:.1%}" if improvement is not None else "—",
            border=True,
        )

    with st.container(border=True):
        st.subheader("Dependence-aware comparison", icon=":material/query_stats:")
        if inference.available:
            with st.container(horizontal=True):
                st.metric(
                    "Mean Heston - BSM loss",
                    f"{inference.mean_loss_difference:.6f}",
                    border=True,
                )
                st.metric(
                    "Bootstrap interval",
                    f"[{inference.confidence_interval_low:.6f}, "
                    f"{inference.confidence_interval_high:.6f}]",
                    border=True,
                )
                st.metric(
                    "P(Heston better)",
                    f"{inference.probability_heston_better:.1%}",
                    border=True,
                )
                st.metric(
                    "Conclusion",
                    inference.preferred_model.title() if inference.conclusive else "Inconclusive",
                    border=True,
                )
            st.caption(
                f"{inference.resamples:,} circular moving-block resamples; "
                f"{inference.evaluation_dates} date clusters; block length "
                f"{inference.block_length}. Negative loss differences favor Heston."
            )
            with st.container(horizontal=True):
                st.metric(
                    "HLN-DM p-value",
                    f"{inference.dm_p_value:.4f}" if inference.dm_p_value is not None else "—",
                    border=True,
                )
                st.metric(
                    "Date wins H / B / ties",
                    f"{inference.heston_win_dates} / {inference.bsm_win_dates} / "
                    f"{inference.tie_dates}",
                    border=True,
                )
                st.metric(
                    "Exact sign-test p-value",
                    (
                        f"{inference.sign_test_p_value:.4f}"
                        if inference.sign_test_p_value is not None
                        else "—"
                    ),
                    border=True,
                )
                st.metric(
                    "HAC lags",
                    inference.hac_lags if inference.hac_lags is not None else "—",
                    border=True,
                )
            st.caption(
                "The HLN-corrected Diebold-Mariano statistic uses a Bartlett HAC variance. "
                "The exact sign test is an independence-based directional sensitivity check; "
                "the block-bootstrap interval remains the primary conclusion rule."
            )
            if not inference.dm_available:
                st.info("HLN-DM diagnostic unavailable: " + str(inference.dm_failure_reason))
            if not inference.sign_test_available:
                st.info(
                    "Sign-test diagnostic unavailable: " + str(inference.sign_test_failure_reason)
                )
        else:
            st.info(
                "Comparison uncertainty is unavailable: " + str(inference.failure_reason),
                icon=":material/info:",
            )

    if not timeline.empty:
        chart_data = timeline.assign(
            evaluation_timestamp=pd.to_datetime(timeline["evaluation_timestamp"])
        )
        timeline_chart = (
            alt.Chart(chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("evaluation_timestamp:T", title="Evaluation snapshot"),
                y=alt.Y("rmse:Q", title="Price RMSE"),
                color=alt.Color("model:N", title="Model"),
                tooltip=[
                    alt.Tooltip("evaluation_timestamp:T", title="Snapshot"),
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip("rmse:Q", title="RMSE", format=".6f"),
                    alt.Tooltip("quote_coverage:Q", title="Inside quote", format=".1%"),
                ],
            )
            .properties(height=300)
        )
        with st.container(border=True):
            st.subheader("Next-snapshot pricing error", icon=":material/timeline:")
            st.altair_chart(
                timeline_chart,
                alt="BSM and Heston next-snapshot price RMSE over evaluation time",
            )

    if not panels.empty:
        heatmap = (
            alt.Chart(panels)
            .mark_rect()
            .encode(
                x=alt.X("moneyness_bucket:N", title="Forward moneyness"),
                y=alt.Y("maturity_bucket:N", title="Maturity"),
                color=alt.Color("rmse:Q", title="Price RMSE"),
                tooltip=[
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip("maturity_bucket:N", title="Maturity"),
                    alt.Tooltip("moneyness_bucket:N", title="Moneyness"),
                    alt.Tooltip("quote_count:Q", title="Quotes"),
                    alt.Tooltip("rmse:Q", title="RMSE", format=".6f"),
                    alt.Tooltip("quote_coverage:Q", title="Inside quote", format=".1%"),
                ],
            )
            .properties(width=280, height=180)
            .facet(column=alt.Column("model:N", title=None))
        )
        with st.container(border=True):
            st.subheader("Maturity and moneyness panel", icon=":material/grid_view:")
            st.altair_chart(
                heatmap,
                alt="BSM and Heston price RMSE by maturity and forward-moneyness bucket",
            )

    with st.container(border=True):
        st.subheader("Auditable model losses", icon=":material/fact_check:")
        st.dataframe(
            summary,
            hide_index=True,
            key="benchmark_model_summary",
            alt="Overall out-of-sample price and quote-band losses by model",
            column_config={
                "rmse": st.column_config.NumberColumn(format="%.6f"),
                "mae": st.column_config.NumberColumn(format="%.6f"),
                "bias": st.column_config.NumberColumn(format="%.6f"),
                "vega_scaled_rmse": st.column_config.NumberColumn(format="%.6f"),
                "vega_scaled_mae": st.column_config.NumberColumn(format="%.6f"),
                "quote_coverage": st.column_config.NumberColumn(format="percent"),
                "mean_quote_band_loss": st.column_config.NumberColumn(format="%.6f"),
            },
        )
        audit_columns = [
            "sequence",
            "timestamp",
            "action",
            "training_start_timestamp",
            "training_end_timestamp",
            "no_lookahead_passed",
            "bsm_rmse",
            "heston_rmse",
            "prediction_hash",
            "audit_hash",
            "failure_reasons",
        ]
        audit = report.audit_frame()
        st.dataframe(
            audit.loc[:, [column for column in audit_columns if column in audit]],
            hide_index=True,
            key="benchmark_audit_steps",
            alt="Chronological training, forecast, loss, and hash-chain audit",
        )
        st.caption(f"Final audit SHA-256: `{report.final_audit_hash or 'unavailable'}`")

    if report.downstream_eligible:
        st.success(
            "Every scored snapshot used only earlier fits, every training surface passed the "
            "arbitrage gate before Heston calibration, and the prediction hash chain verified.",
            icon=":material/verified:",
        )
    else:
        st.error("Benchmark blocked: " + "; ".join(report.failure_reasons))

    with st.container(horizontal=True):
        st.download_button(
            "Download benchmark audit",
            data=benchmark_report_json(report),
            file_name="volforge-model-benchmark.json",
            mime="application/json",
            icon=":material/download:",
        )
        st.download_button(
            "Download prediction panel",
            data=dataframe_to_csv(predictions),
            file_name="volforge-model-predictions.csv",
            mime="text/csv",
            icon=":material/download:",
        )
        st.download_button(
            "Download research bundle",
            data=build_benchmark_research_bundle(report),
            file_name="volforge-benchmark-research-bundle.zip",
            mime="application/zip",
            icon=":material/archive:",
        )

with st.expander("Benchmark interpretation", icon=":material/info:"):
    st.markdown(
        "- Each daily BSM volatility and Heston parameter vector is estimated independently. A "
        "forecast uses component-wise medians from the configured trailing window.\n"
        "- The current snapshot supplies spot, rates, dividends, contract terms, and observed "
        "quotes, but never fitted parameters.\n"
        "- Midpoint RMSE/MAE and bias measure dollar pricing error. Quote-band loss is zero when "
        "the model price lies inside the bid/ask interval and measures only the distance outside.\n"
        "- Model-comparison intervals first average paired losses within each evaluation date, "
        "then resample chronological date blocks. They are deliberately withheld for fewer than "
        "five dates. HLN-DM/HAC and exact date-sign diagnostics are reported separately; the sign "
        "test assumes independent non-tied date signs and is not the primary decision gate.\n"
        "- The synthetic fixture demonstrates chronology and accounting; it is not evidence of "
        "historical or tradable model superiority."
    )
