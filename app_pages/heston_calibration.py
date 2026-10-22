"""Gated Heston calibration and COS-pricing workspace."""

from io import BytesIO

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from volforge.calibration import (
    HestonCalibration,
    calibrate_heston,
    heston_calibration_report_json,
    heston_parameter_frame,
)
from volforge.data import (
    SyntheticChainConfig,
    enrich_option_chain,
    generate_synthetic_chain,
    load_csv,
)
from volforge.models.heston import COSSettings
from volforge.reporting import dataframe_to_csv
from volforge.surfaces import (
    SSVICalibration,
    SVICalibration,
    calibrate_ssvi,
    calibrate_svi_surface,
    prepare_surface_quotes,
)

st.title("Heston calibration", icon=":material/model_training:")
st.caption(
    "Calibrate Heston with Fourier-cosine pricing only after the source SVI/SSVI surface "
    "passes the static-arbitrage gate."
)

DIAGNOSTIC_GRID = np.linspace(-1.25, 1.25, 161)


def fit_chain(
    chain: pd.DataFrame,
    surface_model: str,
    k_min: float,
    k_max: float,
    k_points: int,
    cos_terms: int,
    require_feller: bool,
) -> tuple[pd.DataFrame, SVICalibration | SSVICalibration, HestonCalibration]:
    """Run the presentation-independent surface and Heston pipeline."""
    quotes = prepare_surface_quotes(chain)
    surface = (
        calibrate_svi_surface(quotes, diagnostic_grid=DIAGNOSTIC_GRID)
        if surface_model == "SVI slices"
        else calibrate_ssvi(quotes, diagnostic_grid=DIAGNOSTIC_GRID)
    )
    calibration = calibrate_heston(
        surface,
        log_moneyness=np.linspace(k_min, k_max, k_points),
        settings=COSSettings(terms=cos_terms),
        require_feller=require_feller,
    )
    return quotes, surface, calibration


@st.cache_data(max_entries=10, show_spinner=False)
def calibrate_synthetic(
    config: SyntheticChainConfig,
    surface_model: str,
    k_min: float,
    k_max: float,
    k_points: int,
    cos_terms: int,
    require_feller: bool,
) -> tuple[pd.DataFrame, SVICalibration | SSVICalibration, HestonCalibration]:
    chain = enrich_option_chain(generate_synthetic_chain(config), model="bsm")
    return fit_chain(chain, surface_model, k_min, k_max, k_points, cos_terms, require_feller)


@st.cache_data(max_entries=6, show_spinner=False)
def calibrate_upload(
    content: bytes,
    surface_model: str,
    k_min: float,
    k_max: float,
    k_points: int,
    cos_terms: int,
    require_feller: bool,
) -> tuple[pd.DataFrame, SVICalibration | SSVICalibration, HestonCalibration]:
    chain = load_csv(BytesIO(content), model="bsm")
    return fit_chain(chain, surface_model, k_min, k_max, k_points, cos_terms, require_feller)


with st.sidebar:
    source = st.segmented_control(
        "Calibration data",
        ["Synthetic", "CSV upload"],
        default="Synthetic",
        required=True,
        key="heston_source",
    )
    surface_model = st.segmented_control(
        "Source surface",
        ["SSVI", "SVI slices"],
        default="SSVI",
        required=True,
        key="heston_surface_model",
    )
    with st.form("heston_calibration_settings"):
        if source == "Synthetic":
            base_volatility = st.number_input(
                "Base volatility", min_value=0.01, value=0.22, step=0.01
            )
            skew = st.number_input("Smile skew", value=-0.12, step=0.01)
            curvature = st.number_input("Smile curvature", min_value=0.0, value=0.35, step=0.05)
            noise = st.number_input("Seeded quote noise", min_value=0.0, value=0.0, step=0.005)
            seed = st.number_input("Random seed", min_value=0, value=7, step=1)
        k_min = st.number_input("Minimum log-moneyness", value=-0.4, step=0.05)
        k_max = st.number_input("Maximum log-moneyness", value=0.4, step=0.05)
        k_points = st.number_input("Moneyness points", min_value=5, max_value=15, value=5, step=2)
        cos_terms = st.selectbox("COS terms", [96, 128, 192, 256], index=2)
        require_feller = st.toggle(
            "Require Feller condition",
            value=False,
            help="Parameterizes vol-of-vol so 2κθ - ξ² is non-negative.",
        )
        st.form_submit_button("Calibrate Heston", type="primary", icon=":material/play_arrow:")

if k_min >= k_max:
    st.error("Minimum log-moneyness must be below the maximum.", icon=":material/error:")
    st.stop()

if source == "Synthetic":
    source_input: SyntheticChainConfig | bytes = SyntheticChainConfig(
        base_volatility=base_volatility,
        skew=skew,
        curvature=curvature,
        quote_noise_std=noise,
        seed=int(seed),
    )
else:
    uploaded = st.sidebar.file_uploader("Option-chain CSV", type=["csv"], key="heston_upload")
    if uploaded is None:
        st.info("Upload a CSV file to run gated Heston calibration.", icon=":material/upload:")
        st.stop()
    source_input = uploaded.getvalue()

results_slot = st.container()
try:
    with results_slot.skeleton(height=200):
        if isinstance(source_input, SyntheticChainConfig):
            quotes, surface, calibration = calibrate_synthetic(
                source_input,
                surface_model,
                k_min,
                k_max,
                int(k_points),
                cos_terms,
                require_feller,
            )
        else:
            quotes, surface, calibration = calibrate_upload(
                source_input,
                surface_model,
                k_min,
                k_max,
                int(k_points),
                cos_terms,
                require_feller,
            )
except (ArithmeticError, RuntimeError, ValueError) as exc:
    st.error(f"Heston calibration was blocked or failed: {exc}", icon=":material/block:")
    st.stop()

iv_rmse = calibration.rmse_implied_volatility
with st.container(horizontal=True):
    st.metric("Source expirations", str(quotes["expiration"].nunique()), border=True)
    st.metric("Calibration points", str(len(calibration.points)), border=True)
    st.metric("IV RMSE", f"{iv_rmse:.6f}" if iv_rmse is not None else "Unavailable", border=True)
    st.metric("Price RMSE", f"{calibration.rmse_price:.6e}", border=True)
    st.metric("Function evaluations", str(calibration.function_evaluations), border=True)
    st.metric("Feller margin", f"{calibration.parameters.feller_margin:.6e}", border=True)

if calibration.calibration_eligible:
    st.success(
        "The source surface passed the v0.2 gate and Heston calibration converged with "
        "invertible model prices.",
        icon=":material/verified:",
    )
else:
    st.error("Heston calibration is not eligible: " + "; ".join(calibration.failure_reasons))

if not calibration.parameters.feller_satisfied and not calibration.feller_required:
    st.warning(
        "The Feller condition is not satisfied under the report-only policy. Variance remains "
        "non-negative, but the zero boundary is attainable.",
        icon=":material/warning:",
    )

fit = calibration.evaluated_frame()
iv_chart_data = fit.melt(
    id_vars=["maturity", "log_moneyness"],
    value_vars=["target_implied_volatility", "model_implied_volatility"],
    var_name="series",
    value_name="implied_volatility",
)
iv_chart_data["series"] = iv_chart_data["series"].map(
    {
        "target_implied_volatility": "Surface target",
        "model_implied_volatility": "Heston COS",
    }
)
target_points = (
    alt.Chart(iv_chart_data.loc[iv_chart_data["series"] == "Surface target"])
    .mark_point(filled=True, size=55)
    .encode(
        x=alt.X("log_moneyness:Q", title="Forward log-moneyness log(K/F)"),
        y=alt.Y(
            "implied_volatility:Q",
            title="Black-Scholes implied volatility",
            scale=alt.Scale(zero=False),
        ),
        color=alt.Color("maturity:N", title="Maturity"),
        tooltip=[
            alt.Tooltip("maturity:Q", format=".6f"),
            alt.Tooltip("log_moneyness:Q", format=".4f"),
            alt.Tooltip("implied_volatility:Q", format=".6f"),
        ],
    )
)
model_lines = (
    alt.Chart(iv_chart_data.loc[iv_chart_data["series"] == "Heston COS"])
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("log_moneyness:Q", title="Forward log-moneyness log(K/F)"),
        y=alt.Y(
            "implied_volatility:Q",
            title="Black-Scholes implied volatility",
            scale=alt.Scale(zero=False),
        ),
        color=alt.Color("maturity:N", title="Maturity"),
        tooltip=[
            alt.Tooltip("maturity:Q", format=".6f"),
            alt.Tooltip("log_moneyness:Q", format=".4f"),
            alt.Tooltip("implied_volatility:Q", format=".6f"),
        ],
    )
)
residual_chart = (
    alt.Chart(fit)
    .mark_bar()
    .encode(
        x=alt.X("log_moneyness:O", title="Forward log-moneyness"),
        y=alt.Y("implied_volatility_residual:Q", title="Heston IV minus target IV"),
        color=alt.Color("maturity:N", title="Maturity"),
        tooltip=[
            alt.Tooltip("maturity:Q", format=".6f"),
            alt.Tooltip("log_moneyness:Q", format=".4f"),
            alt.Tooltip("implied_volatility_residual:Q", format=".7f"),
        ],
    )
    .properties(height=330)
)

with st.container(border=True):
    st.subheader("Surface versus Heston fit", icon=":material/show_chart:")
    st.altair_chart(
        (model_lines + target_points).properties(height=390),
        alt="Heston implied-volatility fit against the gated source surface",
    )

left, right = st.columns([1, 2])
with left.container(border=True, height="stretch"):
    st.subheader("Parameters", icon=":material/tune:")
    st.dataframe(
        heston_parameter_frame(calibration),
        hide_index=True,
        alt="Calibrated Heston parameters and diagnostics",
        column_config={
            "kappa": st.column_config.NumberColumn("Kappa", format="%.6f"),
            "theta": st.column_config.NumberColumn("Theta", format="%.8f"),
            "vol_of_vol": st.column_config.NumberColumn("Vol of vol", format="%.6f"),
            "rho": st.column_config.NumberColumn("Rho", format="%.6f"),
            "initial_variance": st.column_config.NumberColumn("Initial variance", format="%.8f"),
            "feller_margin": st.column_config.NumberColumn("Feller margin", format="%.8f"),
            "iv_rmse": st.column_config.NumberColumn("IV RMSE", format="%.8f"),
            "price_rmse": st.column_config.NumberColumn("Price RMSE", format="%.8f"),
        },
    )
with right.container(border=True, height="stretch"):
    st.subheader("Implied-volatility residuals", icon=":material/difference:")
    st.altair_chart(residual_chart, alt="Heston implied-volatility residuals by maturity")

with st.container(horizontal=True):
    st.download_button(
        "Download fit comparison",
        data=dataframe_to_csv(fit),
        file_name="volforge_heston_fit.csv",
        mime="text/csv",
        icon=":material/download:",
        key="heston_fit_download",
    )
    st.download_button(
        "Download calibration audit",
        data=heston_calibration_report_json(calibration),
        file_name="volforge_heston_audit.json",
        mime="application/json",
        icon=":material/download:",
        key="heston_audit_download",
    )

with st.expander("Model and calibration assumptions", icon=":material/info:"):
    st.markdown(
        "- The source is calibrated and validated by the v0.2 SVI/SSVI pipeline first. "
        "A failed surface gate stops Heston calibration.\n"
        "- Calibration uses normalized forward coordinates with `F=1`, `K=exp(k)`, and zero "
        "carry; this preserves the surface's forward log-moneyness geometry.\n"
        "- The objective divides normalized price errors by Black-Scholes vega, approximating "
        "implied-volatility errors without an inner solve during optimization.\n"
        "- Final model prices are inverted independently to report actual implied-volatility "
        "residuals. COS truncation uses corrected Heston cumulants.\n"
        "- The Feller condition is diagnostic by default and a hard parameterization constraint "
        "when requested. Model fit is not trading advice or evidence of executable arbitrage."
    )
