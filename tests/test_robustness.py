import math

import numpy as np
import pytest

from volforge.models.heston import HestonParameters
from volforge.numerics import HestonPathSettings, simulate_heston_paths
from volforge.robustness import (
    HedgeRobustnessSettings,
    hedge_robustness_report_json,
    run_hedge_robustness_gate,
)

PARAMETERS = HestonParameters(1.5, 0.04, 0.4, -0.7, 0.04)


def _quick_settings(**overrides):  # type: ignore[no-untyped-def]
    values = {
        "seeds": (7, 17),
        "path_counts": (6, 10),
        "rebalancing_steps": (2, 3),
        "absolute_tolerance": 10.0,
        "relative_tolerance": 0.0,
        "minimum_volatility_reduction": 0.0,
        "maximum_seed_range_z": 100.0,
        "maximum_clipped_fraction": 1.0,
        "maximum_fallback_fraction": 1.0,
    }
    values.update(overrides)
    return HedgeRobustnessSettings(**values)


def test_physical_spot_drift_is_explicit_and_reproducible() -> None:
    settings = HestonPathSettings(paths=10, steps=4, seed=13)
    neutral = simulate_heston_paths(
        100.0, 1.0, PARAMETERS, 0.03, settings=settings, spot_drift=0.03
    )
    physical = simulate_heston_paths(
        100.0, 1.0, PARAMETERS, 0.03, settings=settings, spot_drift=0.08
    )
    assert np.allclose(physical.variances, neutral.variances)
    assert np.allclose(physical.spots[-1] / neutral.spots[-1], math.exp(0.05))
    with pytest.raises(ValueError, match="spot_drift"):
        simulate_heston_paths(100.0, 1.0, PARAMETERS, settings=settings, spot_drift=float("nan"))


def test_hedge_robustness_is_deterministic_and_auditable() -> None:
    settings = _quick_settings()
    first = run_hedge_robustness_gate(
        "call", 100.0, 100.0, 1.0, PARAMETERS, 0.03, settings=settings
    )
    second = run_hedge_robustness_gate(
        "call", 100.0, 100.0, 1.0, PARAMETERS, 0.03, settings=settings
    )
    assert first == second
    assert hedge_robustness_report_json(first) == hedge_robustness_report_json(second)
    assert first.total_backtest_runs == 12
    assert first.downstream_eligible
    assert first.eligible_strategies
    assert {scenario.name for scenario in first.scenarios} == {
        "q_matched",
        "physical_drift",
        "parameter_misspecification",
        "stressed_costs",
        "combined_stress",
    }
    assert (
        max(
            observation.max_abs_attribution_gap
            for scenario in first.scenarios
            for observation in scenario.strategies
        )
        < 1e-8
    )

    scenario_map = {scenario.name: scenario for scenario in first.scenarios}
    matched = {item.strategy: item for item in scenario_map["q_matched"].strategies}
    costly = {item.strategy: item for item in scenario_map["stressed_costs"].strategies}
    for strategy in matched:
        assert costly[strategy].mean_transaction_cost >= matched[strategy].mean_transaction_cost
    assert scenario_map["parameter_misspecification"].path_parameters != PARAMETERS


def test_robustness_gate_blocks_an_impossible_risk_reduction() -> None:
    report = run_hedge_robustness_gate(
        "put",
        100.0,
        100.0,
        0.5,
        PARAMETERS,
        0.03,
        settings=_quick_settings(minimum_volatility_reduction=1.0),
    )
    assert not report.downstream_eligible
    assert not report.eligible_strategies
    assert report.failure_reasons


@pytest.mark.parametrize(
    "settings",
    [
        lambda: HedgeRobustnessSettings(seeds=(7, 7)),
        lambda: HedgeRobustnessSettings(path_counts=(10,), rebalancing_steps=(4,)),
        lambda: HedgeRobustnessSettings(path_counts=(10, 9), rebalancing_steps=(4, 8)),
        lambda: HedgeRobustnessSettings(path_counts=(10, 20), rebalancing_steps=(8, 4)),
        lambda: HedgeRobustnessSettings(relative_tolerance=1.1),
        lambda: HedgeRobustnessSettings(maximum_clipped_fraction=1.1),
    ],
)
def test_robustness_settings_validation(settings) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        settings()
