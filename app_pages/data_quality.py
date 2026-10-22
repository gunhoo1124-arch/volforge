"""Option-chain quality and put-call parity diagnostics."""

from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st

from volforge.data import (
    SyntheticChainConfig,
    enrich_option_chain,
    generate_synthetic_chain,
    load_csv,
)
from volforge.diagnostics import put_call_parity_residuals, quality_summary
from volforge.reporting import quality_report_markdown

st.title("Data-quality diagnostics", icon=":material/fact_check:")
st.caption("Every failed check remains attached to its source row; nothing is silently discarded.")

source = st.segmented_control(
    "Data source",
    ["Diagnostic synthetic chain", "CSV upload"],
    default="Diagnostic synthetic chain",
    key="quality_source",
)

try:
    if source == "Diagnostic synthetic chain":
        inject_issues = st.toggle(
            "Inject deterministic quality issues",
            value=True,
            help="Adds crossed, negative, stale, above-bound, and duplicate examples.",
            key="quality_inject",
        )
        raw = generate_synthetic_chain(SyntheticChainConfig(inject_quality_issues=inject_issues))
        chain = enrich_option_chain(raw)
    else:
        uploaded = st.file_uploader("Option-chain CSV", type=["csv"], key="quality_upload")
        if uploaded is None:
            st.info("Upload a CSV file to run diagnostics.", icon=":material/upload:")
            st.stop()
        chain = load_csv(BytesIO(uploaded.getvalue()))
except ValueError as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

summary = quality_summary(chain)
with st.container(horizontal=True):
    st.metric("Valid rows", f"{summary['valid_rows']}/{summary['total_rows']}", border=True)
    st.metric("Crossed quotes", str(summary["crossed_quotes"]), border=True)
    st.metric(
        "Bound violations",
        str(summary["below_bound_rows"] + summary["above_bound_rows"]),
        border=True,
    )
    st.metric("Duplicate rows", str(summary["duplicate_rows"]), border=True)
    st.metric("Stale rows", str(summary["stale_rows"]), border=True)

parity = put_call_parity_residuals(chain)
diagnostic_columns = [
    "quote_timestamp",
    "expiration",
    "strike",
    "option_type",
    "bid",
    "ask",
    "midpoint",
    "lower_bound",
    "upper_bound",
    "iv_converged",
    "validation_reasons",
]
invalid = chain.loc[~chain["is_valid"], diagnostic_columns]
left, right = st.columns(2)
with left.container(border=True, height="stretch"):
    st.subheader("Invalid observations", icon=":material/warning:")
    if invalid.empty:
        st.success("No quality failures detected.", icon=":material/check_circle:")
    else:
        st.dataframe(invalid, hide_index=True, alt="Invalid observations and failure reasons")

with right.container(border=True, height="stretch"):
    st.subheader("Put-call parity", icon=":material/balance:")
    if parity.empty:
        st.caption("No unique call-put pairs were available.")
    else:
        parity_chart = (
            alt.Chart(parity)
            .mark_bar()
            .encode(
                x=alt.X("strike:O", title="Strike"),
                y=alt.Y("parity_residual:Q", title="Parity residual"),
                color=alt.condition(
                    "datum.parity_residual >= 0",
                    alt.value("#60A5FA"),
                    alt.value("#F87171"),
                ),
                tooltip=[
                    alt.Tooltip("expiration:T", title="Expiration"),
                    alt.Tooltip("strike:Q", format=".2f"),
                    alt.Tooltip("parity_residual:Q", format=".8f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(parity_chart, alt="Put-call parity residuals by strike")

st.subheader("Quality metrics", icon=":material/analytics:")
metrics_frame = pd.DataFrame({"Metric": list(summary.keys()), "Value": list(summary.values())})
st.dataframe(metrics_frame, hide_index=True, alt="Summary option-chain quality metrics")

st.download_button(
    "Download quality report",
    data=quality_report_markdown(chain).encode("utf-8"),
    file_name="volforge_quality_report.md",
    mime="text/markdown",
    icon=":material/download:",
    key="quality_download",
)

st.warning(
    "Diagnostics indicate data/model consistency only. They are not evidence of an executable "
    "arbitrage and do not account for liquidity, funding, exercise, or transaction costs.",
    icon=":material/info:",
)
