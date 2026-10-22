"""Multi-seed, multi-budget, and misspecification tests for Heston hedges."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from itertools import pairwise

import numpy as np

from volforge.hedging import (
    HEDGE_STRATEGIES,
    HedgeBacktestResult,
    HedgeBacktestSettings,
    HedgeStrategyResult,
    run_heston_hedge_backtest,
)
from volforge.models.common import OptionType
from volforge.models.heston import COSSettings, HestonParameters


@dataclass(frozen=True, slots=True)
class HedgeRobustnessSettings:
    """Seeds, refinement ladder, stresses, and acceptance tolerances."""

    seeds: tuple[int, ...] = (7, 17)
    path_counts: tuple[int, ...] = (10, 20, 40)
    rebalancing_steps: tuple[int, ...] = (4, 6, 8)
    antithetic: bool = True
    stock_cost_bps: float = 1.0
    option_cost_bps: float = 5.0
    cost_stress_multiplier: float = 3.0
    hedge_strike_low: float = 0.9
    hedge_strike_high: float = 1.1
    max_option_position: float = 50.0
    condition_limit: float = 1e6
    physical_annual_return: float = 0.08
    physical_kappa_multiplier: float = 0.75
    physical_theta_multiplier: float = 1.25
    physical_vol_of_vol_multiplier: float = 1.20
    physical_rho_shift: float = -0.10
    absolute_tolerance: float = 0.25
    relative_tolerance: float = 0.05
    minimum_volatility_reduction: float = 0.02
    maximum_seed_range_z: float = 4.0
    maximum_clipped_fraction: float = 0.05
    maximum_fallback_fraction: float = 0.10
    attribution_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("robustness requires at least two distinct seeds")
        if any(not isinstance(seed, int) or seed < 0 for seed in self.seeds):
            raise ValueError("robustness seeds must be non-negative integers")
        if len(self.path_counts) < 2 or len(self.path_counts) != len(self.rebalancing_steps):
            raise ValueError("path and rebalancing ladders require equal lengths of at least two")
        if any(not isinstance(value, int) or value < 2 for value in self.path_counts):
            raise ValueError("path-count ladder values must be integers of at least two")
        if self.antithetic and any(value % 2 for value in self.path_counts):
            raise ValueError("antithetic robustness requires even path counts")
        if any(not isinstance(value, int) or value < 1 for value in self.rebalancing_steps):
            raise ValueError("rebalancing ladder values must be positive integers")
        if any(right <= left for left, right in pairwise(self.path_counts)) or any(
            right <= left for left, right in pairwise(self.rebalancing_steps)
        ):
            raise ValueError("robustness ladders must be strictly increasing")
        non_negative = (
            self.stock_cost_bps,
            self.option_cost_bps,
            self.absolute_tolerance,
            self.relative_tolerance,
            self.minimum_volatility_reduction,
            self.maximum_clipped_fraction,
            self.maximum_fallback_fraction,
            self.attribution_tolerance,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in non_negative):
            raise ValueError("robustness costs and tolerances must be finite and non-negative")
        positive = (
            self.cost_stress_multiplier,
            self.max_option_position,
            self.condition_limit,
            self.maximum_seed_range_z,
            self.physical_kappa_multiplier,
            self.physical_theta_multiplier,
            self.physical_vol_of_vol_multiplier,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("robustness multipliers and limits must be positive and finite")
        if self.condition_limit <= 1.0:
            raise ValueError("condition_limit must be greater than one")
        if not 0.0 < self.hedge_strike_low < 1.0 < self.hedge_strike_high:
            raise ValueError("hedge strike multipliers must bracket one")
        if not math.isfinite(self.physical_annual_return) or not math.isfinite(
            self.physical_rho_shift
        ):
            raise ValueError("physical dynamics inputs must be finite")
        if self.relative_tolerance > 1.0:
            raise ValueError("relative_tolerance must not exceed one")
        if not 0.0 <= self.minimum_volatility_reduction <= 1.0:
            raise ValueError("minimum_volatility_reduction must lie in [0, 1]")
        if self.maximum_clipped_fraction > 1.0 or self.maximum_fallback_fraction > 1.0:
            raise ValueError("diagnostic fractions must not exceed one")


@dataclass(frozen=True, slots=True)
class HedgeAggregate:
    """Pooled path and seed diagnostics for one strategy and one budget."""

    strategy: str
    paths_per_seed: int
    rebalancing_steps: int
    seed_count: int
    samples: int
    mean_pnl: float
    pnl_standard_deviation: float
    pnl_standard_error: float
    pnl_q05: float
    pnl_median: float
    pnl_q95: float
    mean_transaction_cost: float
    mean_turnover: float
    seed_mean_range: float
    risk_reduction_vs_unhedged: float
    max_abs_attribution_gap: float
    clipped_fraction: float
    fallback_fraction: float


@dataclass(frozen=True, slots=True)
class HedgeScenarioResult:
    """Final-budget results under one set of realized dynamics and costs."""

    name: str
    description: str
    path_parameters: HestonParameters
    path_spot_drift: float
    stock_cost_bps: float
    option_cost_bps: float
    strategies: tuple[HedgeAggregate, ...]


@dataclass(frozen=True, slots=True)
class HedgeStrategyRobustness:
    """Numerical and scenario gate for one hedge strategy."""

    strategy: str
    gate_applicable: bool
    refinements: tuple[HedgeAggregate, ...]
    mean_refinement_change: float
    mean_refinement_limit: float
    volatility_refinement_change: float
    volatility_refinement_limit: float
    worst_case_volatility_reduction: float
    maximum_attribution_gap: float
    maximum_seed_range_z: float
    maximum_clipped_fraction: float
    maximum_fallback_fraction: float
    passed: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HedgeRobustnessReport:
    """Auditable robustness assessment across seeds, budgets, and scenarios."""

    target_option_type: str
    target_strike: float
    maturity: float
    pricing_parameters: HestonParameters
    settings: HedgeRobustnessSettings
    total_backtest_runs: int
    scenarios: tuple[HedgeScenarioResult, ...]
    strategies: tuple[HedgeStrategyRobustness, ...]
    eligible_strategies: tuple[str, ...]
    downstream_eligible: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ScenarioSpec:
    name: str
    description: str
    parameters: HestonParameters
    spot_drift: float
    cost_multiplier: float


_DEFAULT_SETTINGS = HedgeRobustnessSettings()
_DEFAULT_COS_SETTINGS = COSSettings(terms=128, truncation_width=16.0)


def _scenario_specs(
    parameters: HestonParameters,
    rate: float,
    dividend_yield: float,
    settings: HedgeRobustnessSettings,
) -> tuple[_ScenarioSpec, ...]:
    physical_rho = parameters.rho + settings.physical_rho_shift
    if not -1.0 < physical_rho < 1.0:
        raise ValueError("physical_rho_shift produces a correlation outside (-1, 1)")
    physical_parameters = HestonParameters(
        kappa=parameters.kappa * settings.physical_kappa_multiplier,
        theta=parameters.theta * settings.physical_theta_multiplier,
        vol_of_vol=parameters.vol_of_vol * settings.physical_vol_of_vol_multiplier,
        rho=physical_rho,
        initial_variance=parameters.initial_variance,
    )
    risk_neutral_drift = rate - dividend_yield
    physical_drift = settings.physical_annual_return - dividend_yield
    return (
        _ScenarioSpec(
            "q_matched",
            "Pricing and realized paths share risk-neutral Heston dynamics.",
            parameters,
            risk_neutral_drift,
            1.0,
        ),
        _ScenarioSpec(
            "physical_drift",
            "Realized spot uses the physical expected return; variance dynamics remain matched.",
            parameters,
            physical_drift,
            1.0,
        ),
        _ScenarioSpec(
            "parameter_misspecification",
            "Physical drift and stressed variance dynamics generate realized paths.",
            physical_parameters,
            physical_drift,
            1.0,
        ),
        _ScenarioSpec(
            "stressed_costs",
            "Risk-neutral paths retain matched parameters while trading costs are multiplied.",
            parameters,
            risk_neutral_drift,
            settings.cost_stress_multiplier,
        ),
        _ScenarioSpec(
            "combined_stress",
            "Physical drift, stressed variance dynamics, and multiplied trading costs apply.",
            physical_parameters,
            physical_drift,
            settings.cost_stress_multiplier,
        ),
    )


def _backtest_settings(
    settings: HedgeRobustnessSettings,
    paths: int,
    steps: int,
    seed: int,
    cost_multiplier: float,
) -> HedgeBacktestSettings:
    return HedgeBacktestSettings(
        paths=paths,
        rebalancing_steps=steps,
        seed=seed,
        antithetic=settings.antithetic,
        stock_cost_bps=settings.stock_cost_bps * cost_multiplier,
        option_cost_bps=settings.option_cost_bps * cost_multiplier,
        hedge_strike_low=settings.hedge_strike_low,
        hedge_strike_high=settings.hedge_strike_high,
        max_option_position=settings.max_option_position,
        condition_limit=settings.condition_limit,
    )


def _option_legs(strategy: str) -> int:
    if strategy in {"delta_gamma", "delta_vega"}:
        return 1
    if strategy == "delta_gamma_vega":
        return 2
    return 0


def _aggregate_one(
    strategy: str,
    results: tuple[HedgeStrategyResult, ...],
    paths: int,
    steps: int,
) -> HedgeAggregate:
    pnl = np.asarray([value for result in results for value in result.pnl], dtype=float)
    seed_means = np.asarray([result.mean_pnl for result in results], dtype=float)
    deviation = float(np.std(pnl, ddof=1))
    option_legs = _option_legs(strategy)
    clipped_denominator = len(results) * paths * steps * option_legs
    fallback_denominator = len(results) * paths * steps if option_legs else 0
    return HedgeAggregate(
        strategy=strategy,
        paths_per_seed=paths,
        rebalancing_steps=steps,
        seed_count=len(results),
        samples=int(pnl.size),
        mean_pnl=float(np.mean(pnl)),
        pnl_standard_deviation=deviation,
        pnl_standard_error=deviation / math.sqrt(pnl.size),
        pnl_q05=float(np.quantile(pnl, 0.05)),
        pnl_median=float(np.median(pnl)),
        pnl_q95=float(np.quantile(pnl, 0.95)),
        mean_transaction_cost=float(np.mean([item.mean_transaction_cost for item in results])),
        mean_turnover=float(np.mean([item.mean_turnover for item in results])),
        seed_mean_range=float(np.ptp(seed_means)),
        risk_reduction_vs_unhedged=0.0,
        max_abs_attribution_gap=max(item.max_abs_attribution_gap for item in results),
        clipped_fraction=(
            sum(item.clipped_rebalances for item in results) / clipped_denominator
            if clipped_denominator
            else 0.0
        ),
        fallback_fraction=(
            sum(item.ill_conditioned_rebalances for item in results) / fallback_denominator
            if fallback_denominator
            else 0.0
        ),
    )


def _aggregate_backtests(backtests: tuple[HedgeBacktestResult, ...]) -> tuple[HedgeAggregate, ...]:
    first = backtests[0]
    aggregates = []
    for strategy in HEDGE_STRATEGIES:
        strategy_results = tuple(
            next(item for item in backtest.strategies if item.strategy == strategy)
            for backtest in backtests
        )
        aggregates.append(
            _aggregate_one(
                strategy,
                strategy_results,
                first.settings.paths,
                first.settings.rebalancing_steps,
            )
        )
    unhedged_deviation = aggregates[0].pnl_standard_deviation
    return tuple(
        replace(
            aggregate,
            risk_reduction_vs_unhedged=(
                1.0 - aggregate.pnl_standard_deviation / unhedged_deviation
                if unhedged_deviation > 0.0
                else 0.0
            ),
        )
        for aggregate in aggregates
    )


def _run_scenario(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    pricing_parameters: HestonParameters,
    rate: float,
    dividend_yield: float,
    settings: HedgeRobustnessSettings,
    cos_settings: COSSettings,
    scenario: _ScenarioSpec,
    paths: int,
    steps: int,
) -> tuple[HedgeBacktestResult, ...]:
    return tuple(
        run_heston_hedge_backtest(
            option_type,
            spot,
            strike,
            maturity,
            pricing_parameters,
            rate,
            dividend_yield,
            settings=_backtest_settings(settings, paths, steps, seed, scenario.cost_multiplier),
            cos_settings=cos_settings,
            path_parameters=scenario.parameters,
            path_spot_drift=scenario.spot_drift,
        )
        for seed in settings.seeds
    )


def _refinement_limits(
    previous: HedgeAggregate,
    final: HedgeAggregate,
    settings: HedgeRobustnessSettings,
) -> tuple[float, float]:
    mean_sampling = 1.959963984540054 * math.sqrt(
        previous.pnl_standard_error**2 + final.pnl_standard_error**2
    )
    previous_std_error = previous.pnl_standard_deviation / math.sqrt(
        max(2.0 * (previous.samples - 1), 1.0)
    )
    final_std_error = final.pnl_standard_deviation / math.sqrt(max(2.0 * (final.samples - 1), 1.0))
    volatility_sampling = 1.959963984540054 * math.sqrt(previous_std_error**2 + final_std_error**2)
    mean_limit = (
        settings.absolute_tolerance
        + settings.relative_tolerance * max(abs(final.mean_pnl), 1.0)
        + mean_sampling
    )
    volatility_limit = (
        settings.absolute_tolerance
        + settings.relative_tolerance * max(final.pnl_standard_deviation, 1.0)
        + volatility_sampling
    )
    return mean_limit, volatility_limit


def _strategy_gate(
    strategy: str,
    refinements: tuple[HedgeAggregate, ...],
    scenarios: tuple[HedgeScenarioResult, ...],
    settings: HedgeRobustnessSettings,
) -> HedgeStrategyRobustness:
    previous = refinements[-2]
    final = refinements[-1]
    mean_change = abs(final.mean_pnl - previous.mean_pnl)
    volatility_change = abs(final.pnl_standard_deviation - previous.pnl_standard_deviation)
    mean_limit, volatility_limit = _refinement_limits(previous, final, settings)
    scenario_results = tuple(
        next(item for item in scenario.strategies if item.strategy == strategy)
        for scenario in scenarios
    )
    worst_reduction = min(item.risk_reduction_vs_unhedged for item in scenario_results)
    all_results = (*refinements, *scenario_results)
    maximum_gap = max(item.max_abs_attribution_gap for item in all_results)
    maximum_seed_range_z = max(
        item.seed_mean_range
        / max(item.pnl_standard_deviation / math.sqrt(item.paths_per_seed), 1e-12)
        for item in all_results
    )
    maximum_clipped = max(item.clipped_fraction for item in all_results)
    maximum_fallback = max(item.fallback_fraction for item in all_results)
    gate_applicable = strategy != "unhedged"
    reasons: list[str] = []
    if gate_applicable and mean_change > mean_limit:
        reasons.append("mean P&L did not stabilize across the final two refinement levels")
    if gate_applicable and volatility_change > volatility_limit:
        reasons.append("P&L volatility did not stabilize across the final two refinement levels")
    if gate_applicable and worst_reduction < settings.minimum_volatility_reduction:
        reasons.append("at least one scenario missed the minimum volatility reduction")
    if maximum_gap > settings.attribution_tolerance:
        reasons.append("P&L attribution exceeded its reconciliation tolerance")
    if gate_applicable and maximum_seed_range_z > settings.maximum_seed_range_z:
        reasons.append("seed-to-seed mean P&L dispersion exceeded its sampling limit")
    if gate_applicable and maximum_clipped > settings.maximum_clipped_fraction:
        reasons.append("option-position clipping exceeded its permitted fraction")
    if gate_applicable and maximum_fallback > settings.maximum_fallback_fraction:
        reasons.append("ill-conditioned hedge fallbacks exceeded their permitted fraction")
    return HedgeStrategyRobustness(
        strategy=strategy,
        gate_applicable=gate_applicable,
        refinements=refinements,
        mean_refinement_change=mean_change,
        mean_refinement_limit=mean_limit,
        volatility_refinement_change=volatility_change,
        volatility_refinement_limit=volatility_limit,
        worst_case_volatility_reduction=worst_reduction,
        maximum_attribution_gap=maximum_gap,
        maximum_seed_range_z=maximum_seed_range_z,
        maximum_clipped_fraction=maximum_clipped,
        maximum_fallback_fraction=maximum_fallback,
        passed=gate_applicable and not reasons,
        failure_reasons=tuple(reasons),
    )


def run_hedge_robustness_gate(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    settings: HedgeRobustnessSettings = _DEFAULT_SETTINGS,
    cos_settings: COSSettings = _DEFAULT_COS_SETTINGS,
) -> HedgeRobustnessReport:
    """Assess hedge stability across seeds, budgets, costs, and path dynamics."""
    kind = OptionType.parse(option_type)
    if any(not math.isfinite(value) for value in (spot, strike, maturity, rate, dividend_yield)):
        raise ValueError("robustness contract inputs must be finite")
    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("robustness requires positive spot, strike, and maturity")
    scenario_specs = _scenario_specs(parameters, rate, dividend_yield, settings)
    baseline = scenario_specs[0]
    refinement_backtests = tuple(
        _run_scenario(
            kind,
            spot,
            strike,
            maturity,
            parameters,
            rate,
            dividend_yield,
            settings,
            cos_settings,
            baseline,
            paths,
            steps,
        )
        for paths, steps in zip(settings.path_counts, settings.rebalancing_steps, strict=True)
    )
    refinement_aggregates = tuple(
        _aggregate_backtests(backtests) for backtests in refinement_backtests
    )
    final_paths = settings.path_counts[-1]
    final_steps = settings.rebalancing_steps[-1]
    scenario_results: list[HedgeScenarioResult] = []
    for index, scenario in enumerate(scenario_specs):
        backtests = (
            refinement_backtests[-1]
            if index == 0
            else _run_scenario(
                kind,
                spot,
                strike,
                maturity,
                parameters,
                rate,
                dividend_yield,
                settings,
                cos_settings,
                scenario,
                final_paths,
                final_steps,
            )
        )
        scenario_results.append(
            HedgeScenarioResult(
                name=scenario.name,
                description=scenario.description,
                path_parameters=scenario.parameters,
                path_spot_drift=scenario.spot_drift,
                stock_cost_bps=settings.stock_cost_bps * scenario.cost_multiplier,
                option_cost_bps=settings.option_cost_bps * scenario.cost_multiplier,
                strategies=_aggregate_backtests(backtests),
            )
        )
    scenarios = tuple(scenario_results)
    strategy_reports = tuple(
        _strategy_gate(
            strategy,
            tuple(level[index] for level in refinement_aggregates),
            scenarios,
            settings,
        )
        for index, strategy in enumerate(HEDGE_STRATEGIES)
    )
    eligible = tuple(report.strategy for report in strategy_reports if report.passed)
    all_reconciled = all(
        aggregate.max_abs_attribution_gap <= settings.attribution_tolerance
        for scenario in scenarios
        for aggregate in scenario.strategies
    )
    failure_reasons: list[str] = []
    if not all_reconciled:
        failure_reasons.append("at least one scenario failed exact P&L reconciliation")
    if not eligible:
        failure_reasons.append("no hedged strategy passed every robustness criterion")
    total_runs = len(settings.seeds) * (len(settings.path_counts) + len(scenario_specs) - 1)
    return HedgeRobustnessReport(
        target_option_type=kind.value,
        target_strike=strike,
        maturity=maturity,
        pricing_parameters=parameters,
        settings=settings,
        total_backtest_runs=total_runs,
        scenarios=scenarios,
        strategies=strategy_reports,
        eligible_strategies=eligible,
        downstream_eligible=all_reconciled and bool(eligible),
        failure_reasons=tuple(failure_reasons),
    )


def hedge_robustness_report_json(report: HedgeRobustnessReport) -> bytes:
    """Serialize a byte-stable robustness audit."""
    return (json.dumps(asdict(report), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
