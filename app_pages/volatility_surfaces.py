"""Constrained SVI and SSVI surface calibration workspace."""

from io import BytesIO

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from volforge.data import (
    SyntheticChainConfig,
    enrich_option_chain,
    generate_synthetic_chain,
    load_csv,
)
from volforge.reporting import dataframe_to_csv
from volforge.surfaces import (
    SSVICalibration,
    SVICalibration,
    calibrate_ssvi,
    calibrate_svi_surface,
    prepare_surface_quotes,
    require_downstream_eligible,
    ssvi_derivatives,
    ssvi_total_variance,
    svi_derivatives,
    svi_total_variance,
)
from volforge.surfaces.arbitrage import density_factor
from volforge.surfaces.common import SurfaceValidationError
from volforge.surfaces.reporting import calibration_report_json, parameter_frame

st.title("Volatility surfaces", icon=":material/stacked_line_chart:")
st.caption(
    "Calibrate in forward log-moneyness and total implied variance, then enforce the "
    "static-arbitrage gate before any future stochastic-volatility handoff."
)

GRID = np.linspace(-1.25, 1.25, 161)


@st.cache_data(max_entries=12, show_spinner=False)
def calibrate_synthetic(
    config: SyntheticChainConfig, model: str
) -> tuple[pd.DataFrame, SVICalibration | SSVICalibration]:
    chain = enrich_option_chain(generate_synthetic_chain(config), model="bsm")
    quotes = prepare_surface_quotes(chain)
    calibration = (
        calibrate_svi_surface(quotes, diagnostic_grid=GRID)
        if model == "SVI slices"
        else calibrate_ssvi(quotes, diagnostic_grid=GRID)
    )
    return quotes, calibration


@st.cache_data(max_entries=8, show_spinner=False)
def calibrate_upload(
    content: bytes, model: str
) -> tuple[pd.DataFrame, SVICalibration | SSVICalibration]:
    chain = load_csv(BytesIO(content), model="bsm")
    quotes = prepare_surface_quotes(chain)
    calibration = (
        calibrate_svi_surface(quotes, diagnostic_grid=GRID)
        if model == "SVI slices"
        else calibrate_ssvi(quotes, diagnostic_grid=GRID)
    )
    return quotes, calibration


with st.sidebar:
    source = st.segmented_control(
        "Surface data", ["Synthetic", "CSV upload"], default="Synthetic", key="surface_source"
    )
    model = st.segmented_control(
        "Calibration model",
        ["SVI slices", "SSVI"],
        default="SSVI",
        key="surface_model",
    )

try:
    if source == "Synthetic":
        with st.sidebar.form("surface_synthetic_settings"):
            base_volatility = st.number_input(
                "Base volatility", min_value=0.01, value=0.22, step=0.01
            )
            skew = st.number_input("Smile skew", value=-0.12, step=0.01)
            curvature = st.number_input("Smile curvature", min_value=0.0, value=0.35, step=0.05)
            noise = st.number_input("Seeded quote noise", min_value=0.0, value=0.0, step=0.005)
            seed = st.number_input("Random seed", min_value=0, value=7, step=1)
            st.form_submit_button("Calibrate", type="primary", icon=":material/play_arrow:")
        config = SyntheticChainConfig(
            base_volatility=base_volatility,
            skew=skew,
            curvature=curvature,
            quote_noise_std=noise,
            seed=int(seed),
        )
        source_input: SyntheticChainConfig | bytes = config
    else:
        uploaded = st.sidebar.file_uploader("Option-chain CSV", type=["csv"], key="surface_upload")
        if uploaded is None:
            st.info("Upload a CSV file to calibrate a surface.", icon=":material/upload:")
            st.stop()
        source_input = uploaded.getvalue()

    results_slot = st.container()
    with results_slot.skeleton(height=180):
        if isinstance(source_input, SyntheticChainConfig):
            quotes, calibration = calibrate_synthetic(source_input, model)
        else:
            quotes, calibration = calibrate_upload(source_input, model)
except (RuntimeError, ValueError) as exc:
    st.error(f"Calibration could not be completed: {exc}", icon=":material/error:")
    st.stop()

minimum_g = (
    min(result.butterfly.minimum_density_factor for result in calibration.slices)
    if isinstance(calibration, SVICalibration)
    else min(report.minimum_density_factor for report in calibration.butterflies)
)
with st.container(horizontal=True):
    st.metric("Expirations", str(quotes["expiration"].nunique()), border=True)
    st.metric("Unique quotes", str(len(quotes)), border=True)
    st.metric("Implied-volatility RMSE", f"{calibration.rmse_implied_volatility:.6f}", border=True)
    st.metric("Minimum density factor", f"{minimum_g:.6f}", border=True)
    st.metric(
        "Minimum calendar spread",
        f"{calibration.calendar.minimum_calendar_spread:.6e}",
        border=True,
    )

try:
    require_downstream_eligible(calibration)
    st.success(
        "Static-arbitrage gate passed. This surface is eligible for the future Heston "
        "calibration stage.",
        icon=":material/verified:",
    )
except SurfaceValidationError as exc:
    st.error(str(exc), icon=":material/block:")

fitted = calibration.evaluated_frame(GRID)
observed = quotes[
    ["expiration", "maturity", "log_moneyness", "total_variance", "implied_volatility"]
].copy()
observed["expiration"] = observed["expiration"].astype(str)
observed["series"] = "Observed"
fitted["expiration"] = fitted["expiration"].astype(str)
fitted["series"] = "Calibrated"
plot_data = pd.concat([observed, fitted], ignore_index=True)

color = alt.Color("expiration:N", title="Expiration")
observed_points = (
    alt.Chart(plot_data.loc[plot_data["series"] == "Observed"])
    .mark_point(size=50, filled=True)
    .encode(
        x=alt.X("log_moneyness:Q", title="Forward log-moneyness log(K/F)"),
        y=alt.Y("total_variance:Q", title="Total implied variance", scale=alt.Scale(zero=False)),
        color=color,
        tooltip=[
            alt.Tooltip("expiration:N", title="Expiration"),
            alt.Tooltip("log_moneyness:Q", format=".4f"),
            alt.Tooltip("total_variance:Q", format=".8f"),
        ],
    )
)
fitted_lines = (
    alt.Chart(plot_data.loc[plot_data["series"] == "Calibrated"])
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("log_moneyness:Q", title="Forward log-moneyness log(K/F)"),
        y=alt.Y("total_variance:Q", title="Total implied variance", scale=alt.Scale(zero=False)),
        color=color,
        tooltip=[
            alt.Tooltip("expiration:N", title="Expiration"),
            alt.Tooltip("log_moneyness:Q", format=".4f"),
            alt.Tooltip("total_variance:Q", format=".8f"),
        ],
    )
)
with st.container(border=True):
    st.subheader("Total implied variance fit", icon=":material/show_chart:")
    st.altair_chart(
        (fitted_lines + observed_points).properties(height=390),
        alt="Observed and calibrated total implied variance by expiration",
    )

density_rows: list[pd.DataFrame] = []
if isinstance(calibration, SVICalibration):
    for result in calibration.slices:
        w = svi_total_variance(GRID, result.parameters)
        first, second = svi_derivatives(GRID, result.parameters)
        density_rows.append(
            pd.DataFrame(
                {
                    "expiration": result.expiration,
                    "log_moneyness": GRID,
                    "density_factor": density_factor(GRID, w, first, second),
                }
            )
        )
else:
    parameters = calibration.parameters
    for expiration, theta in zip(calibration.expirations, parameters.theta, strict=True):
        w = ssvi_total_variance(GRID, theta, parameters.rho, parameters.eta, parameters.gamma)
        first, second = ssvi_derivatives(
            GRID, theta, parameters.rho, parameters.eta, parameters.gamma
        )
        density_rows.append(
            pd.DataFrame(
                {
                    "expiration": expiration,
                    "log_moneyness": GRID,
                    "density_factor": density_factor(GRID, w, first, second),
                }
            )
        )
density_data = pd.concat(density_rows, ignore_index=True)
density_chart = (
    alt.Chart(density_data)
    .mark_line()
    .encode(
        x=alt.X("log_moneyness:Q", title="Forward log-moneyness"),
        y=alt.Y("density_factor:Q", title="Density factor g(k)"),
        color=alt.Color("expiration:N", title="Expiration"),
        tooltip=[
            alt.Tooltip("expiration:N", title="Expiration"),
            alt.Tooltip("log_moneyness:Q", format=".4f"),
            alt.Tooltip("density_factor:Q", format=".6f"),
        ],
    )
    .properties(height=320)
)

maturities = list(calibration.maturities)
calendar_rows: list[pd.DataFrame] = []
for near_index in range(len(maturities) - 1):
    near = maturities[near_index]
    far = maturities[near_index + 1]
    spread = calibration.total_variance(far, GRID) - calibration.total_variance(near, GRID)
    calendar_rows.append(
        pd.DataFrame(
            {
                "maturity_pair": f"{near:.3f} to {far:.3f}y",
                "log_moneyness": GRID,
                "calendar_spread": spread,
            }
        )
    )
calendar_data = pd.concat(calendar_rows, ignore_index=True)
calendar_chart = (
    alt.Chart(calendar_data)
    .mark_line()
    .encode(
        x=alt.X("log_moneyness:Q", title="Forward log-moneyness"),
        y=alt.Y("calendar_spread:Q", title="Far total variance minus near"),
        color=alt.Color("maturity_pair:N", title="Maturity pair"),
        tooltip=[
            alt.Tooltip("maturity_pair:N", title="Maturity pair"),
            alt.Tooltip("log_moneyness:Q", format=".4f"),
            alt.Tooltip("calendar_spread:Q", format=".8f"),
        ],
    )
    .properties(height=320)
)

left, right = st.columns(2)
with left.container(border=True, height="stretch"):
    st.subheader("Butterfly diagnostic", icon=":material/airline_stops:")
    st.altair_chart(density_chart, alt="Risk-neutral density factor across each calibrated smile")
with right.container(border=True, height="stretch"):
    st.subheader("Calendar diagnostic", icon=":material/calendar_month:")
    st.altair_chart(
        calendar_chart,
        alt="Adjacent-maturity total-variance spreads across log-moneyness",
    )

st.subheader("Calibrated parameters", icon=":material/tune:")
st.dataframe(
    parameter_frame(calibration),
    hide_index=True,
    alt="Calibrated volatility-surface parameters and diagnostics",
    column_config={
        "maturity": st.column_config.NumberColumn("Maturity", format="%.6f"),
        "iv_rmse": st.column_config.NumberColumn("IV RMSE", format="%.8f"),
        "minimum_g": st.column_config.NumberColumn("Minimum g(k)", format="%.8f"),
    },
)

with st.container(horizontal=True):
    st.download_button(
        "Download evaluated surface",
        data=dataframe_to_csv(fitted),
        file_name=f"volforge_{model.lower().replace(' ', '_')}_surface.csv",
        mime="text/csv",
        icon=":material/download:",
        key="surface_csv_download",
    )
    st.download_button(
        "Download calibration audit",
        data=calibration_report_json(calibration),
        file_name=f"volforge_{model.lower().replace(' ', '_')}_audit.json",
        mime="application/json",
        icon=":material/download:",
        key="surface_json_download",
    )

with st.expander("Model and gate assumptions", icon=":material/info:"):
    st.markdown(
        "- Coordinates are total implied variance `w = sigma_BS^2 T` and forward "
        "log-moneyness `k = log(K/F)`.\n"
        "- Raw SVI slices constrain positive variance, Lee wing slopes, and sampled `g(k)`.\n"
        "- Power-law SSVI enforces monotone ATM total variance and the Gatheral-Jacquier "
        "calendar and butterfly sufficient conditions on the calibrated maturity domain.\n"
        "- The downstream gate also checks sampled density and calendar spreads. Optimizer "
        "convergence alone is never sufficient.\n"
        "- Diagnostics are model checks, not evidence of an executable arbitrage or trading advice."
    )
