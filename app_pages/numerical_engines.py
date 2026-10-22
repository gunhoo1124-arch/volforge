"""Cross-check Heston COS, Monte Carlo, and finite-difference engines."""

from dataclasses import asdict

import altair as alt
import pandas as pd
import streamlit as st

from volforge.models.heston import COSSettings, HestonParameters, price
from volforge.numerics import (
    ConvergenceSettings,
    HestonConvergenceReport,
    MonteCarloResult,
    MonteCarloSettings,
    PDEGridSettings,
    PDEPriceResult,
    assess_heston_convergence,
    convergence_report_json,
    heston_monte_carlo,
    heston_pde_price,
)

st.title("Numerical engines", icon=":material/function:")
st.caption(
    "Cross-check a Heston European option with independent COS, seeded QE Monte Carlo, "
    "and two-dimensional finite-difference PDE engines, then optionally enforce automatic "
    "refinement and consensus."
)


@st.cache_data(max_entries=12, show_spinner=False)
def run_engines(
    option_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    parameters: HestonParameters,
    cos_terms: int,
    cos_width: float,
    monte_carlo_settings: MonteCarloSettings,
    pde_settings: PDEGridSettings,
) -> tuple[float, MonteCarloResult, PDEPriceResult]:
    """Run three independent Heston valuation paths for one contract."""
    cos_value = price(
        option_type,
        spot,
        strike,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=COSSettings(terms=cos_terms, truncation_width=cos_width),
    )
    monte_carlo = heston_monte_carlo(
        option_type,
        spot,
        strike,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=monte_carlo_settings,
    )
    pde = heston_pde_price(
        option_type,
        spot,
        strike,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=pde_settings,
    )
    return cos_value, monte_carlo, pde


@st.cache_data(max_entries=6, show_spinner=False)
def run_convergence_gate(
    option_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    parameters: HestonParameters,
    settings: ConvergenceSettings,
) -> HestonConvergenceReport:
    """Run and cache the complete automatic refinement ladder."""
    return assess_heston_convergence(
        option_type,
        spot,
        strike,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=settings,
    )


with st.sidebar, st.form("numerical_engine_settings"):
    st.subheader("Contract", icon=":material/contract:")
    option_type = st.segmented_control(
        "Option type", ["call", "put"], default="call", key="engine_option_type"
    )
    spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
    strike = st.number_input("Strike", min_value=0.01, value=100.0, step=1.0)
    maturity = st.number_input("Maturity (ACT/365F years)", min_value=0.0, value=1.0, step=0.25)
    rate = st.number_input("Risk-free rate", value=0.0, step=0.005, format="%.4f")
    dividend_yield = st.number_input("Dividend yield", value=0.0, step=0.005, format="%.4f")

    st.subheader("Heston parameters", icon=":material/tune:")
    kappa = st.number_input("Kappa", min_value=0.0001, value=1.5768, format="%.4f")
    theta = st.number_input("Theta", min_value=0.0001, value=0.0398, format="%.4f")
    vol_of_vol = st.number_input("Vol of vol", min_value=0.0001, value=0.5751, format="%.4f")
    rho = st.number_input("Rho", min_value=-0.999, max_value=0.999, value=-0.5711, format="%.4f")
    initial_variance = st.number_input(
        "Initial variance", min_value=0.0001, value=0.0175, format="%.4f"
    )

    st.subheader("Numerical controls", icon=":material/settings:")
    cos_terms = st.selectbox("Maximum COS terms", [128, 256, 512, 1024], index=1)
    cos_width = st.number_input(
        "Maximum COS truncation width", min_value=10.0, max_value=30.0, value=16.0, step=2.0
    )
    paths = st.selectbox("Monte Carlo paths", [10_000, 20_000, 50_000, 100_000], index=2)
    mc_steps = st.selectbox("Monte Carlo time steps", [32, 64, 126, 252], index=2)
    seed = st.number_input("Monte Carlo seed", min_value=0, value=7, step=1)
    antithetic = st.toggle("Antithetic sampling", value=True)
    control_variate = st.toggle("Discounted-underlying control variate", value=True)
    log_price_points = st.selectbox("PDE log-price points", [61, 81, 101, 121], index=2)
    variance_points = st.selectbox("PDE variance points", [31, 41, 51, 61], index=2)
    pde_time_steps = st.selectbox("PDE time steps", [60, 100, 160, 220], index=2)
    log_price_half_width = st.number_input(
        "PDE log-price half-width", min_value=0.51, value=4.0, step=0.5
    )
    variance_max = st.number_input("PDE maximum variance", min_value=0.01, value=1.0, step=0.1)
    run_gate = st.toggle(
        "Enforce automatic convergence gate",
        value=False,
        key="engine_run_convergence_gate",
    )
    absolute_tolerance = st.number_input(
        "Gate absolute tolerance", min_value=0.0, value=0.03, step=0.01, format="%.4f"
    )
    relative_tolerance = st.number_input(
        "Gate relative tolerance", min_value=0.0, value=0.01, step=0.005, format="%.4f"
    )
    st.form_submit_button(
        "Run cross-check",
        type="primary",
        icon=":material/play_arrow:",
        key="engine_run_cross_check",
    )

parameters = HestonParameters(kappa, theta, vol_of_vol, rho, initial_variance)
mc_settings = MonteCarloSettings(
    paths=int(paths),
    steps=int(mc_steps),
    seed=int(seed),
    antithetic=antithetic,
    control_variate=control_variate,
)
pde_settings = PDEGridSettings(
    log_price_points=int(log_price_points),
    variance_points=int(variance_points),
    time_steps=int(pde_time_steps),
    log_price_half_width=log_price_half_width,
    variance_max=variance_max,
)

results_slot = st.container()
gate_slot = st.container() if run_gate else None
try:
    with results_slot.skeleton(height=220):
        cos_value, monte_carlo, pde = run_engines(
            option_type,
            spot,
            strike,
            maturity,
            rate,
            dividend_yield,
            parameters,
            int(cos_terms),
            cos_width,
            mc_settings,
            pde_settings,
        )
except (ArithmeticError, RuntimeError, ValueError) as exc:
    results_slot.error(f"Numerical cross-check failed: {exc}", icon=":material/error:")
    st.stop()

with results_slot:
    with st.container(horizontal=True):
        st.metric("COS reference", f"{cos_value:,.6f}", border=True)
        st.metric(
            "QE Monte Carlo",
            f"{monte_carlo.price:,.6f}",
            delta=f"{monte_carlo.price - cos_value:+.6f} vs COS",
            border=True,
        )
        st.metric(
            "Finite-difference PDE",
            f"{pde.price:,.6f}",
            delta=f"{pde.price - cos_value:+.6f} vs COS",
            border=True,
        )
        st.metric("Monte Carlo standard error", f"{monte_carlo.standard_error:,.6f}", border=True)

    comparison = pd.DataFrame(
        [
            {
                "Engine": "COS",
                "Price": cos_value,
                "Difference from COS": 0.0,
                "95% lower": cos_value,
                "95% upper": cos_value,
            },
            {
                "Engine": "QE Monte Carlo",
                "Price": monte_carlo.price,
                "Difference from COS": monte_carlo.price - cos_value,
                "95% lower": monte_carlo.confidence_low,
                "95% upper": monte_carlo.confidence_high,
            },
            {
                "Engine": "Finite-difference PDE",
                "Price": pde.price,
                "Difference from COS": pde.price - cos_value,
                "95% lower": pde.price,
                "95% upper": pde.price,
            },
        ]
    )
    points = (
        alt.Chart(comparison)
        .mark_point(filled=True, size=100)
        .encode(
            x=alt.X("Engine:N", title=None),
            y=alt.Y("Price:Q", title="Option price", scale=alt.Scale(zero=False)),
            color=alt.Color("Engine:N", legend=None),
            tooltip=[
                "Engine:N",
                alt.Tooltip("Price:Q", format=".6f"),
                alt.Tooltip("Difference from COS:Q", format="+.6f"),
            ],
        )
    )
    interval = (
        alt.Chart(comparison.loc[comparison["Engine"] == "QE Monte Carlo"])
        .mark_rule(strokeWidth=2)
        .encode(
            x="Engine:N",
            y=alt.Y("95% lower:Q", title="Option price", scale=alt.Scale(zero=False)),
            y2="95% upper:Q",
        )
    )
    reference = (
        alt.Chart(pd.DataFrame({"COS": [cos_value]})).mark_rule(strokeDash=[5, 4]).encode(y="COS:Q")
    )
    with st.container(border=True):
        st.subheader("Engine comparison", icon=":material/compare_arrows:")
        st.altair_chart(
            (interval + points + reference).properties(height=340),
            alt="Heston engine prices with the Monte Carlo 95 percent confidence interval",
        )
        st.dataframe(
            comparison,
            hide_index=True,
            alt="Heston numerical-engine price comparison",
        )

    if monte_carlo.confidence_low <= cos_value <= monte_carlo.confidence_high:
        st.success(
            "The COS reference lies inside the Monte Carlo 95% confidence interval.",
            icon=":material/verified:",
        )
    else:
        st.warning(
            "The COS reference lies outside the reported Monte Carlo interval. Increase paths "
            "and time steps before interpreting the difference.",
            icon=":material/warning:",
        )

    diagnostics = {
        "Heston parameters": asdict(parameters),
        "Monte Carlo": asdict(monte_carlo),
        "PDE": asdict(pde),
    }
    with st.expander("Run diagnostics", icon=":material/diagnosis:"):
        st.json(diagnostics)

if run_gate and gate_slot is not None:
    try:
        convergence_settings = ConvergenceSettings(
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            max_cos_terms=int(cos_terms),
            max_cos_truncation_width=cos_width,
            max_monte_carlo_paths=int(paths),
            max_monte_carlo_steps=int(mc_steps),
            seed=int(seed),
            antithetic=antithetic,
            control_variate=control_variate,
            max_pde_log_price_points=int(log_price_points),
            max_pde_variance_points=int(variance_points),
            max_pde_time_steps=int(pde_time_steps),
            max_pde_log_price_half_width=log_price_half_width,
            max_pde_variance=variance_max,
        )
        with gate_slot.skeleton(height=240):
            convergence = run_convergence_gate(
                option_type,
                spot,
                strike,
                maturity,
                rate,
                dividend_yield,
                parameters,
                convergence_settings,
            )
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        gate_slot.error(f"Convergence gate failed to run: {exc}", icon=":material/error:")
    else:
        with gate_slot:
            st.header("Automatic convergence gate", icon=":material/verified_user:")
            with st.container(horizontal=True):
                st.metric(
                    "Gate status",
                    "Passed" if convergence.downstream_eligible else "Blocked",
                    border=True,
                )
                st.metric("COS reference", f"{convergence.reference_price:,.6f}", border=True)
                st.metric("Consensus price", f"{convergence.consensus_price:,.6f}", border=True)
                st.metric(
                    "Cross-engine spread", f"{convergence.consensus_spread:,.6f}", border=True
                )
                st.metric("Gate tolerance", f"{convergence.tolerance:,.6f}", border=True)

            if convergence.downstream_eligible:
                st.success(
                    "All refinement, sampling, bounds, and cross-engine consensus checks passed. "
                    "This result is eligible for downstream research workflows.",
                    icon=":material/verified:",
                )
            else:
                st.error(
                    "Price blocked from downstream use: " + "; ".join(convergence.failure_reasons),
                    icon=":material/block:",
                )

            history = pd.DataFrame(
                [
                    {
                        "Engine": observation.engine,
                        "Level": observation.level,
                        "Configuration": observation.configuration,
                        "Price": observation.price,
                        "Change from previous": observation.change_from_previous,
                        "Sampling half-width": observation.uncertainty,
                    }
                    for observation in convergence.observations
                ]
            )
            history_chart = (
                alt.Chart(history)
                .mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X("Level:O", title="Automatic refinement level"),
                    y=alt.Y("Price:Q", title="Option price", scale=alt.Scale(zero=False)),
                    color=alt.Color("Engine:N", title="Engine"),
                    tooltip=[
                        "Engine:N",
                        "Level:O",
                        "Configuration:N",
                        alt.Tooltip("Price:Q", format=".6f"),
                        alt.Tooltip("Change from previous:Q", format=".6f"),
                        alt.Tooltip("Sampling half-width:Q", format=".6f"),
                    ],
                )
                .properties(height=340)
            )
            with st.container(border=True):
                st.subheader("Refinement history", icon=":material/trending_up:")
                st.altair_chart(
                    history_chart,
                    alt="COS, Monte Carlo, and PDE prices across three automatic refinement levels",
                )
                st.dataframe(
                    history,
                    hide_index=True,
                    key="numerical_convergence_history",
                    alt="Automatic numerical convergence refinement history",
                )

            st.download_button(
                "Download convergence audit",
                data=convergence_report_json(convergence),
                file_name="volforge_convergence_audit.json",
                mime="application/json",
                icon=":material/download:",
                key="numerical_convergence_download",
            )

with st.expander("Numerical-method assumptions", icon=":material/info:"):
    st.markdown(
        "- COS is the high-accuracy reference for this European payoff.\n"
        "- Monte Carlo uses Andersen quadratic-exponential variance transitions, a martingale "
        "correction, deterministic seeding, antithetic draws, and an optional discounted-underlying "
        "control variate. The interval captures sampling error, not time-discretization bias.\n"
        "- The PDE engine uses a non-uniform log-price/variance grid, central finite differences, "
        "Crank-Nicolson stepping, and Rannacher damping near the payoff kink. Refine all three grid "
        "dimensions and expand the domain when precision matters.\n"
        "- All engines assume European exercise, continuous rates and dividends, and constant "
        "risk-neutral Heston parameters. The automatic gate is conditional on its finite maximum "
        "budgets and is not a proof of asymptotic convergence. Results are research diagnostics, "
        "not trading advice."
    )
