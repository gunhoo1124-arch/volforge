"""Option-chain upload, synthesis, enrichment, and smile exploration."""

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
from volforge.reporting import dataframe_to_csv

st.title("Option-chain explorer", icon=":material/table_chart:")
st.caption("Upload the documented CSV schema or generate a reproducible offline chain.")


@st.cache_data(max_entries=16, show_spinner=False)
def build_synthetic(config: SyntheticChainConfig, model: str) -> pd.DataFrame:
    return enrich_option_chain(generate_synthetic_chain(config), model=model)  # type: ignore[arg-type]


@st.cache_data(max_entries=8, show_spinner=False)
def parse_upload(content: bytes, model: str) -> pd.DataFrame:
    return load_csv(BytesIO(content), model=model)  # type: ignore[arg-type]


with st.sidebar:
    source = st.segmented_control(
        "Data source", ["Synthetic", "CSV upload"], default="Synthetic", key="explorer_source"
    )
    model_label = st.segmented_control(
        "Implied-volatility model",
        ["BSM", "Bachelier"],
        default="BSM",
        key="explorer_model",
    )
    model = "bsm" if model_label == "BSM" else "bachelier"

try:
    if source == "Synthetic":
        with st.sidebar.form("explorer_synthetic_settings"):
            spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
            base_vol = st.number_input(
                "Base lognormal volatility", min_value=0.001, value=0.22, step=0.01
            )
            skew = st.number_input("Smile skew", value=-0.12, step=0.01)
            curvature = st.number_input("Smile curvature", min_value=0.0, value=0.35, step=0.05)
            spread = st.number_input("Bid/ask spread", min_value=0.0, value=0.10, step=0.01)
            noise = st.number_input("Seeded quote noise", min_value=0.0, value=0.0, step=0.01)
            seed = st.number_input("Random seed", min_value=0, value=7, step=1)
            st.form_submit_button("Generate", type="primary", icon=":material/refresh:")
        raw_config = SyntheticChainConfig(
            spot=spot,
            base_volatility=base_vol,
            skew=skew,
            curvature=curvature,
            bid_ask_spread=spread,
            quote_noise_std=noise,
            seed=int(seed),
        )
        chain = build_synthetic(raw_config, model)
        source_caption = f"Deterministic synthetic chain · seed {int(seed)}"
    else:
        uploaded = st.sidebar.file_uploader("Option-chain CSV", type=["csv"], key="explorer_upload")
        if uploaded is None:
            st.info(
                "Upload a CSV file to begin, or switch to the synthetic source.",
                icon=":material/upload:",
            )
            st.stop()
        chain = parse_upload(uploaded.getvalue(), model)
        source_caption = uploaded.name
except ValueError as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

st.caption(source_caption)
valid = chain.loc[chain["is_valid"] & chain["iv_converged"]].copy()
valid["expiration_label"] = valid["expiration"].dt.strftime("%Y-%m-%d")
expirations = valid["expiration_label"].dropna().drop_duplicates().tolist()
selected_expirations = st.multiselect(
    "Expirations", expirations, default=expirations, key="explorer_expirations"
)
selected = valid.loc[valid["expiration_label"].isin(selected_expirations)]

with st.container(horizontal=True):
    st.metric("Rows", f"{len(chain):,}", border=True)
    st.metric("Valid", f"{int(chain['is_valid'].sum()):,}", border=True)
    st.metric("Invalid", f"{int((~chain['is_valid']).sum()):,}", border=True)
    st.metric("IV solutions", f"{int(chain['iv_converged'].sum()):,}", border=True)

if selected.empty:
    st.warning("No valid implied-volatility observations match the current filter.")
else:
    y_title = "Lognormal implied volatility" if model == "bsm" else "Normal implied volatility"
    smile_chart = (
        alt.Chart(selected)
        .mark_line(point=True)
        .encode(
            x=alt.X("strike:Q", title="Strike"),
            y=alt.Y("implied_volatility:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=alt.Color("expiration_label:N", title="Expiration"),
            strokeDash=alt.StrokeDash("option_type:N", title="Option type"),
            tooltip=[
                alt.Tooltip("expiration_label:N", title="Expiration"),
                alt.Tooltip("option_type:N", title="Type"),
                alt.Tooltip("strike:Q", format=".2f"),
                alt.Tooltip("implied_volatility:Q", format=".6f"),
            ],
        )
        .properties(height=390)
    )
    with st.container(border=True):
        st.subheader("Implied-volatility smiles", icon=":material/show_chart:")
        st.altair_chart(smile_chart, alt="Implied volatility by strike and expiration")

display_columns = [
    "expiration",
    "strike",
    "option_type",
    "bid",
    "ask",
    "midpoint",
    "maturity_years",
    "implied_volatility",
    "is_valid",
    "validation_reasons",
]
st.dataframe(
    chain[display_columns],
    hide_index=True,
    alt="Enriched option-chain observations",
    column_config={
        "strike": st.column_config.NumberColumn("Strike", format="%.4f", pinned=True),
        "bid": st.column_config.NumberColumn("Bid", format="%.6f"),
        "ask": st.column_config.NumberColumn("Ask", format="%.6f"),
        "midpoint": st.column_config.NumberColumn("Midpoint", format="%.6f"),
        "maturity_years": st.column_config.NumberColumn("Maturity (years)", format="%.6f"),
        "implied_volatility": st.column_config.NumberColumn("Implied volatility", format="%.6f"),
    },
)

invalid = chain.loc[~chain["is_valid"], display_columns]
with st.expander(
    f"Invalid-row explanations ({len(invalid)})",
    icon=":material/warning:",
):
    if invalid.empty:
        st.caption("No invalid observations in the current chain.")
    else:
        st.dataframe(invalid, hide_index=True, alt="Invalid rows and their validation reasons")

st.download_button(
    "Download enriched chain",
    data=dataframe_to_csv(chain),
    file_name="volforge_enriched_chain.csv",
    mime="text/csv",
    icon=":material/download:",
    key="explorer_download",
)

with st.expander("CSV schema and conventions", icon=":material/info:"):
    st.code(
        "quote_timestamp,expiration,strike,option_type,bid,ask,underlying_spot,"
        "risk_free_rate,dividend_yield",
        language="text",
    )
    st.markdown(
        "Timestamps use ISO 8601 and are parsed as UTC. Maturity is ACT/365F. Rates and yields "
        "are continuous annual decimals. Bad rows are retained with explicit validation reasons."
    )
