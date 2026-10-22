"""American-option cross-check workspace."""

from __future__ import annotations

from dataclasses import asdict

import altair as alt
import pandas as pd
import streamlit as st

from volforge.american import (
    BinomialSettings,
    LSMSettings,
    american_binomial_price,
    heston_american_pde_price,
    heston_lsm_price,
)
from volforge.models.heston import HestonParameters
from volforge.numerics import PDEGridSettings


@st.cache_data(max_entries=16, show_spinner=False)
def run_american_cross_check(
    option_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    parameters: HestonParameters,
    lsm_settings: LSMSettings,
    pde_settings: PDEGridSettings,
    benchmark_volatility: float,
    binomial_steps: int,
):
    """Run independent American engines behind a bounded cache."""
    lsm = heston_lsm_price(
        option_type,
        spot,
        strike,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=lsm_settings,
    )
    pde = heston_american_pde_price(
        option_type,
        spot,
        strike,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=pde_settings,
    )
    binomial = american_binomial_price(
        option_type,
        spot,
        strike,
        maturity,
        benchmark_volatility,
        rate,
        dividend_yield,
        settings=BinomialSettings(binomial_steps),
    )
    return lsm, pde, binomial


st.title("American options")
st.caption(
    "Cross-check early exercise with Heston Longstaff-Schwartz simulation, an "
    "Ikonen-Toivanen obstacle PDE, and a constant-volatility CRR benchmark."
)

with st.form("american_option_controls"):
    st.subheader("Contract and model", icon=":material/tune:")
    contract_columns = st.columns(3)
    with contract_columns[0]:
        option_type = st.segmented_control(
            "Option type", ["put", "call"], default="put", key="american_option_type"
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
        benchmark_volatility = st.number_input(
            "CRR benchmark volatility", min_value=0.0001, value=0.20, step=0.01
        )

    st.subheader("Numerical budgets", icon=":material/memory:")
    budget_columns = st.columns(3)
    with budget_columns[0]:
        paths = st.selectbox("LSM paths", [4_000, 10_000, 20_000, 50_000], index=0)
        lsm_steps = st.selectbox("LSM exercise dates", [24, 32, 64, 126], index=0)
        seed = st.number_input("LSM seed", min_value=0, value=7, step=1)
    with budget_columns[1]:
        log_price_points = st.selectbox("PDE log-price points", [41, 51, 61, 81], index=1)
        variance_points = st.selectbox("PDE variance points", [21, 25, 31, 41], index=1)
        pde_time_steps = st.selectbox("PDE time steps", [30, 40, 60, 100], index=1)
    with budget_columns[2]:
        log_price_half_width = st.number_input(
            "PDE log-price half-width", min_value=0.51, value=4.0, step=0.5
        )
        variance_max = st.number_input("PDE maximum variance", min_value=0.01, value=1.0, step=0.1)
        binomial_steps = st.selectbox("CRR steps", [250, 500, 1_000, 2_000], index=1)
    st.form_submit_button(
        "Run American cross-check",
        type="primary",
        icon=":material/play_arrow:",
        key="american_run_cross_check",
    )

parameters = HestonParameters(kappa, theta, vol_of_vol, rho, initial_variance)
lsm_settings = LSMSettings(paths=int(paths), steps=int(lsm_steps), seed=int(seed))
pde_settings = PDEGridSettings(
    log_price_points=int(log_price_points),
    variance_points=int(variance_points),
    time_steps=int(pde_time_steps),
    log_price_half_width=log_price_half_width,
    variance_max=variance_max,
)
results_slot = st.container()
try:
    with results_slot.skeleton(height=260):
        lsm, pde, binomial = run_american_cross_check(
            str(option_type),
            spot,
            strike,
            maturity,
            rate,
            dividend_yield,
            parameters,
            lsm_settings,
            pde_settings,
            benchmark_volatility,
            int(binomial_steps),
        )
except (ArithmeticError, RuntimeError, ValueError) as exc:
    results_slot.error(f"American cross-check failed: {exc}", icon=":material/error:")
    st.stop()

with results_slot:
    with st.container(horizontal=True):
        st.metric(
            "Heston LSM",
            f"{lsm.price:,.6f}",
            delta=f"{lsm.early_exercise_premium:+.6f} vs path European",
            border=True,
        )
        st.metric(
            "Heston obstacle PDE",
            f"{pde.price:,.6f}",
            delta=f"{pde.early_exercise_premium:+.6f} vs PDE European",
            border=True,
        )
        st.metric("Heston engine spread", f"{abs(lsm.price - pde.price):,.6f}", border=True)
        st.metric(
            "CRR benchmark",
            f"{binomial.price:,.6f}",
            delta=f"{binomial.early_exercise_premium:+.6f} vs BSM European",
            border=True,
        )

    comparison = pd.DataFrame(
        [
            {
                "Engine": "Heston LSM",
                "American price": lsm.price,
                "European price": lsm.european_price,
            },
            {
                "Engine": "Heston obstacle PDE",
                "American price": pde.price,
                "European price": pde.european_price,
            },
            {
                "Engine": "BSM CRR benchmark",
                "American price": binomial.price,
                "European price": binomial.european_price,
            },
        ]
    )
    comparison["Early-exercise premium"] = (
        comparison["American price"] - comparison["European price"]
    )
    with st.container(border=True):
        st.subheader("Price comparison", icon=":material/compare_arrows:")
        st.dataframe(
            comparison,
            hide_index=True,
            key="american_price_comparison",
            alt="American and European values from the three pricing engines",
        )

    boundary_rows = [
        {"Exercise time": point.exercise_time, "Underlying": point.underlying, "Engine": engine}
        for engine, points in (
            ("Heston LSM", lsm.exercise_boundary),
            ("Heston obstacle PDE", pde.exercise_boundary),
            ("BSM CRR", binomial.exercise_boundary),
        )
        for point in points
    ]
    if boundary_rows:
        boundary_frame = pd.DataFrame(boundary_rows)
        chart = (
            alt.Chart(boundary_frame)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("Exercise time:Q", title="Time from valuation (years)"),
                y=alt.Y(
                    "Underlying:Q",
                    title="Exercise-boundary underlying",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color("Engine:N", title="Engine"),
                tooltip=[
                    "Engine:N",
                    alt.Tooltip("Exercise time:Q", format=".4f"),
                    alt.Tooltip("Underlying:Q", format=".4f"),
                ],
            )
            .properties(height=360)
        )
        with st.container(border=True):
            st.subheader("Exercise boundary", icon=":material/show_chart:")
            st.altair_chart(
                chart,
                alt="Estimated exercise boundaries from simulation, PDE, and binomial methods",
            )

    if lsm.confidence_low <= pde.price <= lsm.confidence_high:
        st.success(
            "The obstacle-PDE value lies inside the LSM sampling interval.",
            icon=":material/verified:",
        )
    else:
        st.warning(
            "The Heston engines differ beyond the LSM sampling interval. Increase paths, exercise "
            "dates, and PDE resolution before interpreting the premium.",
            icon=":material/warning:",
        )
    with st.expander("Run diagnostics", icon=":material/diagnosis:"):
        st.json(
            {
                "Heston parameters": asdict(parameters),
                "Longstaff-Schwartz": asdict(lsm),
                "Obstacle PDE": asdict(pde),
                "CRR benchmark": asdict(binomial),
            }
        )

with st.expander("Method assumptions", icon=":material/info:"):
    st.markdown(
        "- LSM regresses discounted future cash flow on spot and variance basis functions. Its "
        "interval is conditional on the fitted policy and does not include exercise-grid bias.\n"
        "- The obstacle PDE applies Ikonen-Toivanen splitting to the existing non-uniform Heston "
        "grid with Crank-Nicolson stepping and Rannacher damping.\n"
        "- The CRR value is a Black-Scholes constant-volatility benchmark, not a Heston price.\n"
        "- American calls with no continuous dividends and non-negative rates are never exercised "
        "early; the LSM engine enforces this identity. All results require refinement studies."
    )
