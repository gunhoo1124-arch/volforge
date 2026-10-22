"""Independent Heston Greek-estimator workspace."""

from __future__ import annotations

from dataclasses import asdict

import altair as alt
import pandas as pd
import streamlit as st

from volforge.greeks import (
    FiniteDifferenceSettings,
    MonteCarloGreekSettings,
    heston_finite_difference_greeks,
    heston_monte_carlo_greeks,
)
from volforge.models.heston import COSSettings, HestonParameters
from volforge.numerics import PDEGridSettings


@st.cache_data(max_entries=16, show_spinner=False)
def run_greek_cross_check(
    option_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    parameters: HestonParameters,
    engine: str,
    finite_settings: FiniteDifferenceSettings,
    cos_settings: COSSettings,
    pde_settings: PDEGridSettings,
    monte_carlo_settings: MonteCarloGreekSettings,
):
    """Run deterministic and simulation estimators behind a bounded cache."""
    deterministic = heston_finite_difference_greeks(
        option_type,
        spot,
        strike,
        maturity,
        parameters,
        rate,
        dividend_yield,
        engine=engine,
        settings=finite_settings,
        cos_settings=cos_settings,
        pde_settings=pde_settings,
    )
    monte_carlo = heston_monte_carlo_greeks(
        option_type,
        spot,
        strike,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=monte_carlo_settings,
    )
    return deterministic, monte_carlo


st.title("Heston Greeks")
st.caption(
    "Cross-check five-point COS/PDE bumps against pathwise, conditional likelihood-ratio, "
    "and common-random-number Monte Carlo estimators."
)

with st.form("heston_greek_controls"):
    st.subheader("Contract and model", icon=":material/tune:")
    contract_columns = st.columns(3)
    with contract_columns[0]:
        option_type = st.segmented_control(
            "Option type", ["call", "put"], default="call", key="greek_option_type"
        )
        spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
        strike = st.number_input("Strike", min_value=0.01, value=100.0, step=1.0)
        maturity = st.number_input(
            "Maturity (ACT/365F years)", min_value=0.01, value=1.0, step=0.25
        )
    with contract_columns[1]:
        rate = st.number_input("Continuous rate", value=0.03, step=0.005, format="%.4f")
        dividend_yield = st.number_input(
            "Continuous dividend yield", value=0.0, step=0.005, format="%.4f"
        )
        kappa = st.number_input("Mean reversion (kappa)", min_value=0.001, value=1.5768)
        theta = st.number_input("Long-run variance (theta)", min_value=0.0001, value=0.0398)
    with contract_columns[2]:
        vol_of_vol = st.number_input("Volatility of variance", min_value=0.001, value=0.5751)
        rho = st.number_input(
            "Spot/variance correlation", min_value=-0.999, max_value=0.999, value=-0.5711
        )
        initial_variance = st.number_input("Initial variance", min_value=0.0001, value=0.0175)
        reference_engine = st.segmented_control(
            "Deterministic reference", ["COS", "PDE"], default="COS", key="greek_engine"
        )

    st.subheader("Numerical budgets", icon=":material/memory:")
    budget_columns = st.columns(3)
    with budget_columns[0]:
        paths = st.selectbox("Monte Carlo paths", [4_000, 10_000, 20_000, 50_000], index=0)
        mc_steps = st.selectbox("Monte Carlo time steps", [24, 48, 64, 126], index=0)
        seed = st.number_input("Monte Carlo seed", min_value=0, value=7, step=1)
    with budget_columns[1]:
        cos_terms = st.selectbox("COS terms", [128, 256, 512, 1_024], index=1)
        cos_width = st.number_input(
            "COS truncation width", min_value=5.0, max_value=30.0, value=16.0, step=2.0
        )
        spot_bump = st.number_input(
            "Spot relative bump", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f"
        )
    with budget_columns[2]:
        log_price_points = st.selectbox("PDE log-price points", [41, 61, 81, 121], index=0)
        variance_points = st.selectbox("PDE variance points", [21, 31, 41, 61], index=0)
        pde_time_steps = st.selectbox("PDE time steps", [30, 60, 100, 180], index=0)
        variance_max = st.number_input("PDE maximum variance", min_value=0.01, value=1.0, step=0.1)

    bump_columns = st.columns(3)
    with bump_columns[0]:
        volatility_bump = st.number_input(
            "Volatility relative bump",
            min_value=0.0001,
            max_value=0.1,
            value=0.01,
            format="%.4f",
        )
    with bump_columns[1]:
        maturity_bump = st.number_input(
            "Maturity relative bump",
            min_value=0.0001,
            max_value=0.1,
            value=0.01,
            format="%.4f",
        )
    with bump_columns[2]:
        rate_bump = st.number_input(
            "Absolute rate bump",
            min_value=0.00001,
            max_value=0.1,
            value=0.0001,
            format="%.5f",
        )
    st.form_submit_button(
        "Run Greek cross-check",
        type="primary",
        icon=":material/play_arrow:",
        key="greek_run_cross_check",
    )

parameters = HestonParameters(kappa, theta, vol_of_vol, rho, initial_variance)
finite_settings = FiniteDifferenceSettings(
    spot_relative_bump=spot_bump,
    initial_volatility_relative_bump=volatility_bump,
    maturity_relative_bump=maturity_bump,
    rate_bump=rate_bump,
)
cos_settings = COSSettings(int(cos_terms), cos_width)
pde_settings = PDEGridSettings(
    log_price_points=int(log_price_points),
    variance_points=int(variance_points),
    time_steps=int(pde_time_steps),
    variance_max=variance_max,
)
monte_carlo_settings = MonteCarloGreekSettings(
    paths=int(paths),
    steps=int(mc_steps),
    seed=int(seed),
    initial_volatility_relative_bump=volatility_bump,
    maturity_relative_bump=maturity_bump,
)

results_slot = st.container()
try:
    with results_slot.skeleton(height=260):
        deterministic, monte_carlo = run_greek_cross_check(
            str(option_type),
            spot,
            strike,
            maturity,
            rate,
            dividend_yield,
            parameters,
            str(reference_engine).lower(),
            finite_settings,
            cos_settings,
            pde_settings,
            monte_carlo_settings,
        )
except (ArithmeticError, RuntimeError, ValueError) as exc:
    results_slot.error(f"Greek cross-check failed: {exc}", icon=":material/error:")
    st.stop()

greek_names = ("delta", "gamma", "vega", "theta", "rho")
rows = []
for name in greek_names:
    reference = getattr(deterministic, name)
    estimate = getattr(monte_carlo, name)
    difference = estimate.value - reference.value
    uncertainty = max(estimate.standard_error, reference.estimated_error, 1e-12)
    rows.append(
        {
            "Greek": name.capitalize(),
            "Deterministic": reference.value,
            "Bump discrepancy": reference.estimated_error,
            "Monte Carlo": estimate.value,
            "MC standard error": estimate.standard_error,
            "95% lower": estimate.confidence_low,
            "95% upper": estimate.confidence_high,
            "Difference": difference,
            "Standardized difference": abs(difference) / uncertainty,
            "MC method": estimate.method,
        }
    )
comparison = pd.DataFrame(rows)

with results_slot:
    with st.container(horizontal=True):
        for row in rows:
            st.metric(
                row["Greek"],
                f"{row['Deterministic']:,.6f}",
                delta=f"{row['Difference']:+.6f} MC difference",
                border=True,
            )

    with st.container(border=True):
        st.subheader("Estimator comparison", icon=":material/compare_arrows:")
        st.dataframe(
            comparison,
            hide_index=True,
            key="greek_estimator_comparison",
            alt="Deterministic and Monte Carlo Heston Greek estimates",
        )

    discrepancy_chart = (
        alt.Chart(comparison)
        .mark_bar()
        .encode(
            x=alt.X("Greek:N", title=None),
            y=alt.Y(
                "Standardized difference:Q", title="Absolute difference / reported uncertainty"
            ),
            color=alt.condition(
                alt.datum["Standardized difference"] <= 1.959963984540054,
                alt.value("#2e7d32"),
                alt.value("#c62828"),
            ),
            tooltip=[
                "Greek:N",
                alt.Tooltip("Deterministic:Q", format=".6f"),
                alt.Tooltip("Monte Carlo:Q", format=".6f"),
                alt.Tooltip("Standardized difference:Q", format=".3f"),
            ],
        )
        .properties(height=320)
    )
    threshold = (
        alt.Chart(pd.DataFrame({"threshold": [1.959963984540054]}))
        .mark_rule(strokeDash=[5, 4])
        .encode(y="threshold:Q")
    )
    with st.container(border=True):
        st.subheader("Method disagreement", icon=":material/monitoring:")
        st.altair_chart(
            discrepancy_chart + threshold,
            alt="Greek estimator differences scaled by reported numerical or sampling uncertainty",
        )

    compatible = all(
        row["95% lower"] - getattr(deterministic, name).estimated_error
        <= getattr(deterministic, name).value
        <= row["95% upper"] + getattr(deterministic, name).estimated_error
        for name, row in zip(greek_names, rows, strict=True)
    )
    if compatible:
        st.success(
            "All deterministic estimates lie inside the corresponding Monte Carlo intervals.",
            icon=":material/verified:",
        )
    else:
        st.warning(
            "At least one deterministic estimate lies outside its Monte Carlo interval. Increase "
            "paths and time steps, then vary bump sizes and deterministic resolution.",
            icon=":material/warning:",
        )

    with st.expander("Run diagnostics", icon=":material/diagnosis:"):
        st.json(
            {
                "Heston parameters": asdict(parameters),
                "Deterministic": asdict(deterministic),
                "Monte Carlo": asdict(monte_carlo),
            }
        )

with st.expander("Estimator conventions", icon=":material/info:"):
    st.markdown(
        "- Delta and rho use pathwise estimators; gamma uses the conditional Gaussian "
        "likelihood-ratio score. A second likelihood-ratio delta is retained in diagnostics.\n"
        "- Vega is the response to a parallel absolute shift in `sqrt(v0)` and `sqrt(theta)`.\n"
        "- Theta is calendar decay per year, `-dV/dT`. Rho is per absolute 1.00 rate move.\n"
        "- Monte Carlo vega and theta use paired common random numbers. Their intervals omit "
        "time-step and bump bias. The deterministic discrepancy is empirical, not an error bound."
    )
