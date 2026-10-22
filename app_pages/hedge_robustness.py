"""Multi-scenario Heston hedge-robustness workspace."""

from __future__ import annotations

from dataclasses import asdict

import altair as alt
import pandas as pd
import streamlit as st

from volforge.models.heston import COSSettings, HestonParameters
from volforge.robustness import (
    HedgeRobustnessSettings,
    hedge_robustness_report_json,
    run_hedge_robustness_gate,
)


@st.cache_data(max_entries=8, show_spinner=False)
def run_robustness_lab(
    option_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    parameters: HestonParameters,
    settings: HedgeRobustnessSettings,
    cos_settings: COSSettings,
):
    """Run the complete robustness gate behind a bounded cache."""
    return run_hedge_robustness_gate(
        option_type,
        spot,
        strike,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=settings,
        cos_settings=cos_settings,
    )


STRATEGY_LABELS = {
    "unhedged": "Unhedged",
    "delta": "Delta",
    "delta_gamma": "Delta-gamma",
    "delta_vega": "Delta-vega",
    "delta_gamma_vega": "Delta-gamma-vega",
}
SCENARIO_LABELS = {
    "q_matched": "Q matched",
    "physical_drift": "Physical drift",
    "parameter_misspecification": "Parameter misspecification",
    "stressed_costs": "Stressed costs",
    "combined_stress": "Combined stress",
}
BUDGETS = {
    "Quick": ((10, 20), (4, 6)),
    "Standard": ((10, 20, 40), (4, 6, 8)),
    "Research": ((20, 50, 100), (4, 8, 12)),
}

st.title("Hedge robustness")
st.caption(
    "Gate Heston hedge policies across independent seeds, joint simulation/rebalance refinement, "
    "physical dynamics, parameter misspecification, and transaction-cost stress."
)

with st.form("hedge_robustness_controls"):
    st.subheader("Contract and pricing model", icon=":material/tune:")
    contract_columns = st.columns(3)
    with contract_columns[0]:
        option_type = st.segmented_control(
            "Target option type", ["call", "put"], default="call", key="robust_option_type"
        )
        spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
        strike = st.number_input("Target strike", min_value=0.01, value=100.0, step=1.0)
        maturity = st.number_input(
            "Maturity (ACT/365F years)", min_value=0.01, value=1.0, step=0.25
        )
    with contract_columns[1]:
        rate = st.number_input("Continuous rate", value=0.03, step=0.005, format="%.4f")
        dividend_yield = st.number_input(
            "Continuous dividend yield", value=0.0, step=0.005, format="%.4f"
        )
        kappa = st.number_input("Pricing mean reversion", min_value=0.001, value=1.5)
        theta = st.number_input("Pricing long-run variance", min_value=0.0001, value=0.04)
    with contract_columns[2]:
        vol_of_vol = st.number_input("Pricing volatility of variance", min_value=0.001, value=0.4)
        rho = st.number_input(
            "Pricing spot/variance correlation",
            min_value=-0.999,
            max_value=0.999,
            value=-0.7,
        )
        initial_variance = st.number_input("Initial variance", min_value=0.0001, value=0.04)
        cos_terms = st.selectbox("COS terms", [64, 128, 256, 512], index=1)

    st.subheader("Sampling and hedge design", icon=":material/experiment:")
    design_columns = st.columns(3)
    with design_columns[0]:
        budget_profile = st.segmented_control(
            "Refinement budget", list(BUDGETS), default="Quick", key="robust_budget"
        )
        seed_count = st.selectbox("Independent seeds", [2, 3, 5], index=0)
        base_seed = st.number_input("Base seed", min_value=0, value=7, step=1)
    with design_columns[1]:
        stock_cost_bps = st.number_input(
            "Stock cost (bps of notional)", min_value=0.0, value=1.0, step=0.5
        )
        option_cost_bps = st.number_input(
            "Option cost (bps of premium)", min_value=0.0, value=5.0, step=1.0
        )
        cost_stress_multiplier = st.number_input(
            "Cost stress multiplier", min_value=1.0, value=3.0, step=0.5
        )
    with design_columns[2]:
        hedge_strike_low = st.number_input(
            "Low call strike / target strike",
            min_value=0.1,
            max_value=0.99,
            value=0.9,
            step=0.05,
        )
        hedge_strike_high = st.number_input(
            "High call strike / target strike",
            min_value=1.01,
            value=1.1,
            step=0.05,
        )
        max_option_position = st.number_input(
            "Option position cap", min_value=0.1, value=50.0, step=5.0
        )

    st.subheader("Physical dynamics and acceptance", icon=":material/science:")
    stress_columns = st.columns(3)
    with stress_columns[0]:
        physical_annual_return = st.number_input(
            "Physical annual total return", value=0.08, step=0.01, format="%.3f"
        )
        physical_kappa_multiplier = st.number_input(
            "Physical kappa / pricing kappa", min_value=0.05, value=0.75, step=0.05
        )
        physical_theta_multiplier = st.number_input(
            "Physical theta / pricing theta", min_value=0.05, value=1.25, step=0.05
        )
    with stress_columns[1]:
        physical_vol_of_vol_multiplier = st.number_input(
            "Physical vol-of-vol / pricing vol-of-vol", min_value=0.05, value=1.2, step=0.05
        )
        physical_rho_shift = st.number_input(
            "Physical correlation shift",
            min_value=-1.0,
            max_value=1.0,
            value=-0.1,
            step=0.05,
        )
        minimum_volatility_reduction = st.number_input(
            "Minimum P&L volatility reduction",
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.01,
        )
    with stress_columns[2]:
        absolute_tolerance = st.number_input(
            "Absolute refinement tolerance", min_value=0.0, value=0.25, step=0.05
        )
        relative_tolerance = st.number_input(
            "Relative refinement tolerance",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
        )
        maximum_diagnostic_fraction = st.number_input(
            "Maximum clip/fallback fraction",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
        )
    st.form_submit_button(
        "Run robustness gate",
        type="primary",
        icon=":material/play_arrow:",
        key="robust_run_gate",
    )

path_counts, rebalancing_steps = BUDGETS[str(budget_profile)]
seed_offsets = (0, 10, 22, 36, 52)
seeds = tuple(int(base_seed) + offset for offset in seed_offsets[: int(seed_count)])
parameters = HestonParameters(kappa, theta, vol_of_vol, rho, initial_variance)
settings = HedgeRobustnessSettings(
    seeds=seeds,
    path_counts=path_counts,
    rebalancing_steps=rebalancing_steps,
    stock_cost_bps=stock_cost_bps,
    option_cost_bps=option_cost_bps,
    cost_stress_multiplier=cost_stress_multiplier,
    hedge_strike_low=hedge_strike_low,
    hedge_strike_high=hedge_strike_high,
    max_option_position=max_option_position,
    physical_annual_return=physical_annual_return,
    physical_kappa_multiplier=physical_kappa_multiplier,
    physical_theta_multiplier=physical_theta_multiplier,
    physical_vol_of_vol_multiplier=physical_vol_of_vol_multiplier,
    physical_rho_shift=physical_rho_shift,
    absolute_tolerance=absolute_tolerance,
    relative_tolerance=relative_tolerance,
    minimum_volatility_reduction=minimum_volatility_reduction,
    maximum_seed_range_z=4.0,
    maximum_clipped_fraction=maximum_diagnostic_fraction,
    maximum_fallback_fraction=maximum_diagnostic_fraction,
)
cos_settings = COSSettings(int(cos_terms), 16.0)

results_slot = st.container()
try:
    with results_slot.skeleton(height=300):
        report = run_robustness_lab(
            str(option_type),
            spot,
            strike,
            maturity,
            rate,
            dividend_yield,
            parameters,
            settings,
            cos_settings,
        )
except (ArithmeticError, RuntimeError, ValueError) as exc:
    results_slot.error(f"Robustness gate failed to run: {exc}", icon=":material/error:")
    st.stop()

gate_rows = []
refinement_rows = []
for strategy in report.strategies:
    final = strategy.refinements[-1]
    gate_rows.append(
        {
            "Strategy": STRATEGY_LABELS[strategy.strategy],
            "Status": (
                "Comparator"
                if not strategy.gate_applicable
                else "Passed"
                if strategy.passed
                else "Blocked"
            ),
            "Final mean P&L": final.mean_pnl,
            "Final P&L volatility": final.pnl_standard_deviation,
            "Worst-case volatility reduction": strategy.worst_case_volatility_reduction,
            "Mean refinement change": strategy.mean_refinement_change,
            "Mean refinement limit": strategy.mean_refinement_limit,
            "Volatility refinement change": strategy.volatility_refinement_change,
            "Volatility refinement limit": strategy.volatility_refinement_limit,
            "Maximum seed range z": strategy.maximum_seed_range_z,
            "Maximum clipped fraction": strategy.maximum_clipped_fraction,
            "Maximum fallback fraction": strategy.maximum_fallback_fraction,
            "Failure reasons": "; ".join(strategy.failure_reasons),
        }
    )
    for level, observation in enumerate(strategy.refinements, start=1):
        refinement_rows.append(
            {
                "Strategy": STRATEGY_LABELS[strategy.strategy],
                "Level": level,
                "Budget": f"{observation.paths_per_seed} paths x "
                f"{observation.rebalancing_steps} dates",
                "P&L volatility": observation.pnl_standard_deviation,
                "Mean P&L": observation.mean_pnl,
            }
        )

scenario_rows = [
    {
        "Scenario": SCENARIO_LABELS[scenario.name],
        "Strategy": STRATEGY_LABELS[observation.strategy],
        "Mean P&L": observation.mean_pnl,
        "P&L volatility": observation.pnl_standard_deviation,
        "Volatility reduction": observation.risk_reduction_vs_unhedged,
        "Mean transaction cost": observation.mean_transaction_cost,
        "Seed mean range": observation.seed_mean_range,
        "Clipped fraction": observation.clipped_fraction,
        "Fallback fraction": observation.fallback_fraction,
    }
    for scenario in report.scenarios
    for observation in scenario.strategies
]
gate_frame = pd.DataFrame(gate_rows)
refinement_frame = pd.DataFrame(refinement_rows)
scenario_frame = pd.DataFrame(scenario_rows)

with results_slot:
    eligible_label = ", ".join(STRATEGY_LABELS[name] for name in report.eligible_strategies)
    passing = [item for item in report.strategies if item.passed]
    worst_eligible_reduction = min(
        (item.worst_case_volatility_reduction for item in passing), default=0.0
    )
    with st.container(horizontal=True):
        st.metric(
            "Gate status",
            "Passed" if report.downstream_eligible else "Blocked",
            border=True,
        )
        st.metric("Eligible strategies", str(len(report.eligible_strategies)), border=True)
        st.metric("Backtest runs", str(report.total_backtest_runs), border=True)
        st.metric(
            "Worst eligible risk reduction",
            f"{worst_eligible_reduction:.1%}",
            border=True,
        )
    st.caption(f"Eligible: {eligible_label or 'none'}")

    with st.container(border=True):
        st.subheader("Strategy gate", icon=":material/verified_user:")
        st.dataframe(
            gate_frame,
            hide_index=True,
            key="robust_strategy_gate",
            alt="Hedge-strategy robustness decisions and their numerical diagnostics",
            column_config={
                "Worst-case volatility reduction": st.column_config.NumberColumn(format="percent"),
                "Maximum clipped fraction": st.column_config.NumberColumn(format="percent"),
                "Maximum fallback fraction": st.column_config.NumberColumn(format="percent"),
            },
        )

    refinement_chart = (
        alt.Chart(refinement_frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("Level:O", title="Joint refinement level"),
            y=alt.Y("P&L volatility:Q", title="P&L standard deviation"),
            color=alt.Color("Strategy:N", title="Strategy"),
            tooltip=[
                "Strategy:N",
                "Budget:N",
                alt.Tooltip("Mean P&L:Q", format=".4f"),
                alt.Tooltip("P&L volatility:Q", format=".4f"),
            ],
        )
        .properties(height=340)
    )
    with st.container(border=True):
        st.subheader("Joint refinement", icon=":material/trending_up:")
        st.altair_chart(
            refinement_chart,
            alt="P&L volatility by hedge strategy as path and rebalance budgets increase",
        )

    hedged_scenarios = scenario_frame[scenario_frame["Strategy"] != "Unhedged"]
    scenario_chart = (
        alt.Chart(hedged_scenarios)
        .mark_bar()
        .encode(
            x=alt.X("Scenario:N", sort=list(SCENARIO_LABELS.values()), title=None),
            y=alt.Y("Volatility reduction:Q", title="Reduction versus unhedged"),
            xOffset="Strategy:N",
            color=alt.Color("Strategy:N", title="Strategy"),
            tooltip=[
                "Scenario:N",
                "Strategy:N",
                alt.Tooltip("Volatility reduction:Q", format=".1%"),
                alt.Tooltip("Mean P&L:Q", format=".4f"),
                alt.Tooltip("Mean transaction cost:Q", format=".4f"),
            ],
        )
        .properties(height=360)
    )
    zero_rule = alt.Chart(pd.DataFrame({"zero": [0.0]})).mark_rule().encode(y="zero:Q")
    with st.container(border=True):
        st.subheader("Scenario resilience", icon=":material/stress_management:")
        st.altair_chart(
            scenario_chart + zero_rule,
            alt="Hedge P&L volatility reduction versus unhedged across five realized scenarios",
        )
        st.dataframe(
            scenario_frame,
            hide_index=True,
            key="robust_scenario_results",
            alt="Pooled seed results for every robustness scenario and strategy",
            column_config={
                "Volatility reduction": st.column_config.NumberColumn(format="percent"),
                "Clipped fraction": st.column_config.NumberColumn(format="percent"),
                "Fallback fraction": st.column_config.NumberColumn(format="percent"),
            },
        )

    if report.downstream_eligible:
        st.success(
            "At least one hedged strategy passed refinement, scenario-risk, stability, and exact "
            "reconciliation criteria.",
            icon=":material/verified:",
        )
    else:
        st.error("Robustness gate blocked: " + "; ".join(report.failure_reasons))

    audit = hedge_robustness_report_json(report)
    st.download_button(
        "Download robustness audit",
        data=audit,
        file_name="volforge-hedge-robustness.json",
        mime="application/json",
        icon=":material/download:",
    )
    diagnostics = st.expander("Full diagnostics", icon=":material/diagnosis:")
    with diagnostics:
        st.json(asdict(report))

with st.expander("Gate interpretation", icon=":material/info:"):
    st.markdown(
        "- Pricing and Greeks always use the displayed risk-neutral parameters. Realized paths "
        "may instead use a physical expected return and stressed variance parameters.\n"
        "- The refinement ladder increases Heston path count and rebalance frequency together. "
        "Final changes are compared with configured tolerances plus sampling allowances.\n"
        "- A strategy must reduce terminal P&L volatility versus the unhedged comparator in every "
        "scenario and remain below clipping, fallback, and attribution limits.\n"
        "- Passing is evidence only for these synthetic scenarios and budgets. It is not an "
        "out-of-sample result or permission to trade."
    )
