"""Self-financing Heston hedging backtests with exact P&L attribution."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from volforge.models.common import OptionType
from volforge.models.heston import COSSettings, HestonParameters, cos_prices
from volforge.numerics.monte_carlo import HestonPathSettings, simulate_heston_paths

FloatArray = NDArray[np.float64]

HEDGE_STRATEGIES = (
    "unhedged",
    "delta",
    "delta_gamma",
    "delta_vega",
    "delta_gamma_vega",
)

_PRICE = 0
_DELTA = 1
_GAMMA = 2
_VEGA = 3
_THETA = 4
_VARIANCE_FLOOR = 1e-8


@dataclass(frozen=True, slots=True)
class HedgeBacktestSettings:
    """Simulation, rebalancing, cost, and finite-difference controls."""

    paths: int = 100
    rebalancing_steps: int = 12
    seed: int = 7
    antithetic: bool = True
    stock_cost_bps: float = 1.0
    option_cost_bps: float = 5.0
    hedge_strike_low: float = 0.9
    hedge_strike_high: float = 1.1
    max_option_position: float = 50.0
    spot_relative_bump: float = 1e-3
    instantaneous_volatility_relative_bump: float = 1e-2
    maturity_relative_bump: float = 1e-2
    condition_limit: float = 1e6

    def __post_init__(self) -> None:
        if not isinstance(self.paths, int) or self.paths < 2:
            raise ValueError("hedging paths must be an integer of at least two")
        if self.antithetic and self.paths % 2:
            raise ValueError("antithetic hedging requires an even path count")
        if not isinstance(self.rebalancing_steps, int) or self.rebalancing_steps < 1:
            raise ValueError("rebalancing_steps must be a positive integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("hedging seed must be a non-negative integer")
        costs = (self.stock_cost_bps, self.option_cost_bps)
        if any(not math.isfinite(value) or not 0.0 <= value <= 10_000.0 for value in costs):
            raise ValueError("transaction costs must be finite and between 0 and 10,000 bps")
        if not (
            math.isfinite(self.hedge_strike_low)
            and math.isfinite(self.hedge_strike_high)
            and 0.0 < self.hedge_strike_low < 1.0 < self.hedge_strike_high
        ):
            raise ValueError("hedge strike multipliers must bracket one")
        if not math.isfinite(self.max_option_position) or self.max_option_position <= 0.0:
            raise ValueError("max_option_position must be positive and finite")
        bumps = (
            self.spot_relative_bump,
            self.instantaneous_volatility_relative_bump,
            self.maturity_relative_bump,
        )
        if any(not math.isfinite(value) or not 0.0 < value <= 0.1 for value in bumps):
            raise ValueError("hedging finite-difference bumps must lie in (0, 0.1]")
        if not math.isfinite(self.condition_limit) or self.condition_limit <= 1.0:
            raise ValueError("condition_limit must be finite and greater than one")


@dataclass(frozen=True, slots=True)
class HedgePnLAttribution:
    """Mean terminal P&L attribution per short target option."""

    delta: float
    gamma: float
    vega: float
    theta: float
    financing: float
    transaction_cost: float
    residual: float
    total: float


@dataclass(frozen=True, slots=True)
class HedgeStrategyResult:
    """Distribution, cost, attribution, and hedge-stability diagnostics."""

    strategy: str
    mean_pnl: float
    pnl_standard_deviation: float
    pnl_standard_error: float
    pnl_q05: float
    pnl_median: float
    pnl_q95: float
    mean_transaction_cost: float
    mean_turnover: float
    mean_attribution: HedgePnLAttribution
    max_abs_attribution_gap: float
    clipped_rebalances: int
    ill_conditioned_rebalances: int
    mean_condition_number: float | None
    max_condition_number: float | None
    pnl: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class HedgeBacktestResult:
    """A common-path comparison of all supported hedge strategies."""

    target_option_type: str
    target_strike: float
    maturity: float
    initial_option_price: float
    hedge_strikes: tuple[float, float]
    settings: HedgeBacktestSettings
    pricing_parameters: HestonParameters
    path_parameters: HestonParameters
    path_spot_drift: float
    vega_variable: str
    theta_convention: str
    strategies: tuple[HedgeStrategyResult, ...]


@dataclass(frozen=True, slots=True)
class HestonRiskMetrics:
    """One contract's COS price and finite-difference hedge sensitivities."""

    price: float
    delta: float
    gamma: float
    instantaneous_vega: float
    calendar_theta: float


_DEFAULT_SETTINGS = HedgeBacktestSettings()
_DEFAULT_COS_SETTINGS = COSSettings(terms=128, truncation_width=16.0)


def _validate_contract(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
) -> None:
    values = (spot, strike, maturity, rate, dividend_yield)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("hedging contract inputs must be finite")
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if maturity <= 0.0:
        raise ValueError("hedging maturity must be positive")


def _cos_metrics(
    option_type: OptionType,
    spot: float,
    strikes: FloatArray,
    maturity: float,
    parameters: HestonParameters,
    rate: float,
    dividend_yield: float,
    settings: HedgeBacktestSettings,
    cos_settings: COSSettings,
) -> FloatArray:
    """Return price, delta, gamma, instantaneous vega, and calendar theta."""
    spot_bump = min(max(spot * settings.spot_relative_bump, 1e-5), 0.2 * spot)
    initial_volatility = math.sqrt(parameters.initial_variance)
    volatility_bump = min(
        max(initial_volatility * settings.instantaneous_volatility_relative_bump, 1e-5),
        0.2 * initial_volatility,
    )
    maturity_bump = min(max(maturity * settings.maturity_relative_bump, 1e-5), 0.2 * maturity)

    base = cos_prices(
        option_type,
        spot,
        strikes,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=cos_settings,
    )
    spot_minus = cos_prices(
        option_type,
        spot - spot_bump,
        strikes,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=cos_settings,
    )
    spot_plus = cos_prices(
        option_type,
        spot + spot_bump,
        strikes,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=cos_settings,
    )

    volatility_minus = cos_prices(
        option_type,
        spot,
        strikes,
        maturity,
        replace(parameters, initial_variance=(initial_volatility - volatility_bump) ** 2),
        rate,
        dividend_yield,
        settings=cos_settings,
    )
    volatility_plus = cos_prices(
        option_type,
        spot,
        strikes,
        maturity,
        replace(parameters, initial_variance=(initial_volatility + volatility_bump) ** 2),
        rate,
        dividend_yield,
        settings=cos_settings,
    )
    maturity_minus = cos_prices(
        option_type,
        spot,
        strikes,
        maturity - maturity_bump,
        parameters,
        rate,
        dividend_yield,
        settings=cos_settings,
    )
    maturity_plus = cos_prices(
        option_type,
        spot,
        strikes,
        maturity + maturity_bump,
        parameters,
        rate,
        dividend_yield,
        settings=cos_settings,
    )

    metrics = np.empty((strikes.size, 5), dtype=float)
    metrics[:, _PRICE] = base
    metrics[:, _DELTA] = (spot_plus - spot_minus) / (2.0 * spot_bump)
    metrics[:, _GAMMA] = (spot_plus - 2.0 * base + spot_minus) / spot_bump**2
    metrics[:, _VEGA] = (volatility_plus - volatility_minus) / (2.0 * volatility_bump)
    metrics[:, _THETA] = (maturity_minus - maturity_plus) / (2.0 * maturity_bump)
    if np.any(~np.isfinite(metrics)):
        raise ArithmeticError("hedging risk metrics are non-finite")
    return metrics


def heston_risk_metrics(
    option_type: str | OptionType,
    spot: float,
    strikes: FloatArray,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    settings: HedgeBacktestSettings = _DEFAULT_SETTINGS,
    cos_settings: COSSettings = _DEFAULT_COS_SETTINGS,
) -> tuple[HestonRiskMetrics, ...]:
    """Return model risk metrics used to size listed-contract hedges.

    Vega is sensitivity to instantaneous volatility ``sqrt(v)`` while all
    other Heston parameters remain fixed. Prices are model values; callers
    can independently use observed market marks for self-financing ledgers.
    """
    kind = OptionType.parse(option_type)
    strike_array = np.asarray(strikes, dtype=float)
    if strike_array.ndim != 1 or strike_array.size < 1:
        raise ValueError("strikes must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(strike_array)) or np.any(strike_array <= 0.0):
        raise ValueError("strikes must be positive and finite")
    _validate_contract(spot, float(strike_array[0]), maturity, rate, dividend_yield)
    values = _cos_metrics(
        kind,
        spot,
        strike_array,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings,
        cos_settings,
    )
    return tuple(
        HestonRiskMetrics(
            price=float(row[_PRICE]),
            delta=float(row[_DELTA]),
            gamma=float(row[_GAMMA]),
            instantaneous_vega=float(row[_VEGA]),
            calendar_theta=float(row[_THETA]),
        )
        for row in values
    )


def _state_metrics(
    option_type: OptionType,
    spot: float,
    variance: float,
    target_strike: float,
    hedge_strikes: FloatArray,
    maturity: float,
    parameters: HestonParameters,
    rate: float,
    dividend_yield: float,
    settings: HedgeBacktestSettings,
    cos_settings: COSSettings,
) -> FloatArray:
    current_parameters = replace(parameters, initial_variance=max(variance, _VARIANCE_FLOOR))
    if option_type is OptionType.CALL:
        strikes = np.concatenate(([target_strike], hedge_strikes))
        return _cos_metrics(
            option_type,
            spot,
            strikes,
            maturity,
            current_parameters,
            rate,
            dividend_yield,
            settings,
            cos_settings,
        )
    target = _cos_metrics(
        option_type,
        spot,
        np.asarray([target_strike]),
        maturity,
        current_parameters,
        rate,
        dividend_yield,
        settings,
        cos_settings,
    )
    hedges = _cos_metrics(
        OptionType.CALL,
        spot,
        hedge_strikes,
        maturity,
        current_parameters,
        rate,
        dividend_yield,
        settings,
        cos_settings,
    )
    return np.vstack((target, hedges))


def _precompute_metrics(
    option_type: OptionType,
    spots: FloatArray,
    variances: FloatArray,
    target_strike: float,
    hedge_strikes: FloatArray,
    maturity: float,
    parameters: HestonParameters,
    rate: float,
    dividend_yield: float,
    settings: HedgeBacktestSettings,
    cos_settings: COSSettings,
) -> tuple[FloatArray, FloatArray]:
    steps = settings.rebalancing_steps
    paths = settings.paths
    metrics = np.empty((steps, paths, 3, 5), dtype=float)
    values = np.empty((steps + 1, paths, 3), dtype=float)
    initial = _state_metrics(
        option_type,
        float(spots[0, 0]),
        float(variances[0, 0]),
        target_strike,
        hedge_strikes,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings,
        cos_settings,
    )
    metrics[0, :, :, :] = initial
    values[0, :, :] = initial[:, _PRICE]
    for step in range(1, steps):
        remaining_maturity = maturity * (steps - step) / steps
        for path in range(paths):
            state = _state_metrics(
                option_type,
                float(spots[step, path]),
                float(variances[step, path]),
                target_strike,
                hedge_strikes,
                remaining_maturity,
                parameters,
                rate,
                dividend_yield,
                settings,
                cos_settings,
            )
            metrics[step, path] = state
            values[step, path] = state[:, _PRICE]

    terminal_spots = spots[-1]
    if option_type is OptionType.CALL:
        values[-1, :, 0] = np.maximum(terminal_spots - target_strike, 0.0)
    else:
        values[-1, :, 0] = np.maximum(target_strike - terminal_spots, 0.0)
    values[-1, :, 1:] = np.maximum(terminal_spots[:, None] - hedge_strikes, 0.0)
    return metrics, values


def _clip_option_positions(positions: FloatArray, limit: float) -> tuple[FloatArray, int]:
    clipped = np.clip(positions, -limit, limit)
    return clipped, int(np.count_nonzero(clipped != positions))


def _desired_positions(
    strategy: str,
    metrics: FloatArray,
    settings: HedgeBacktestSettings,
) -> tuple[FloatArray, FloatArray, int, int, list[float]]:
    paths = metrics.shape[0]
    stock = np.zeros(paths, dtype=float)
    options = np.zeros((paths, 2), dtype=float)
    clipped_count = 0
    ill_conditioned = 0
    condition_numbers: list[float] = []
    if strategy == "unhedged":
        return stock, options, clipped_count, ill_conditioned, condition_numbers

    target = metrics[:, 0]
    hedges = metrics[:, 1:]
    if strategy == "delta":
        stock = target[:, _DELTA].copy()
        return stock, options, clipped_count, ill_conditioned, condition_numbers

    if strategy in {"delta_gamma", "delta_vega"}:
        hedge_index = 0 if strategy == "delta_gamma" else 1
        exposure_index = _GAMMA if strategy == "delta_gamma" else _VEGA
        denominator = hedges[:, hedge_index, exposure_index]
        floor = 1e-10
        stable = np.abs(denominator) > floor
        raw = np.zeros(paths, dtype=float)
        raw[stable] = target[stable, exposure_index] / denominator[stable]
        ill_conditioned = int(np.count_nonzero(~stable))
        clipped, clipped_count = _clip_option_positions(raw, settings.max_option_position)
        options[:, hedge_index] = clipped
        stock = target[:, _DELTA] - np.sum(options * hedges[:, :, _DELTA], axis=1)
        return stock, options, clipped_count, ill_conditioned, condition_numbers

    if strategy != "delta_gamma_vega":
        raise ValueError(f"unknown hedge strategy: {strategy}")

    raw = np.zeros((paths, 2), dtype=float)
    for path in range(paths):
        gamma_scale = max(
            abs(target[path, _GAMMA]),
            float(np.max(np.abs(hedges[path, :, _GAMMA]))),
            1e-10,
        )
        vega_scale = max(
            abs(target[path, _VEGA]),
            float(np.max(np.abs(hedges[path, :, _VEGA]))),
            1e-10,
        )
        system = np.asarray(
            [
                hedges[path, :, _GAMMA] / gamma_scale,
                hedges[path, :, _VEGA] / vega_scale,
            ]
        )
        target_exposure = np.asarray(
            [target[path, _GAMMA] / gamma_scale, target[path, _VEGA] / vega_scale]
        )
        condition = float(np.linalg.cond(system))
        if not math.isfinite(condition):
            condition = settings.condition_limit * 10.0
        condition_numbers.append(condition)
        solved = False
        if condition <= settings.condition_limit:
            try:
                raw[path] = np.linalg.solve(system, target_exposure)
            except np.linalg.LinAlgError:
                solved = False
            else:
                solved = True
        if solved:
            continue
        ill_conditioned += 1
        raw[path] = np.linalg.lstsq(system, target_exposure, rcond=1.0 / settings.condition_limit)[
            0
        ]

    options, clipped_count = _clip_option_positions(raw, settings.max_option_position)
    stock = target[:, _DELTA] - np.sum(options * hedges[:, :, _DELTA], axis=1)
    return stock, options, clipped_count, ill_conditioned, condition_numbers


def _strategy_result(
    strategy: str,
    spots: FloatArray,
    variances: FloatArray,
    metrics: FloatArray,
    values: FloatArray,
    rate: float,
    dividend_yield: float,
    settings: HedgeBacktestSettings,
    maturity: float,
) -> HedgeStrategyResult:
    paths = settings.paths
    steps = settings.rebalancing_steps
    time_step = maturity / steps
    cash_growth = math.exp(rate * time_step)
    dividend_growth = math.exp(dividend_yield * time_step) - 1.0
    stock_cost_rate = settings.stock_cost_bps * 1e-4
    option_cost_rate = settings.option_cost_bps * 1e-4

    cash = values[0, :, 0].copy()
    stock_position = np.zeros(paths, dtype=float)
    option_position = np.zeros((paths, 2), dtype=float)
    transaction_cost = np.zeros(paths, dtype=float)
    turnover = np.zeros(paths, dtype=float)
    attribution = {
        name: np.zeros(paths, dtype=float)
        for name in ("delta", "gamma", "vega", "theta", "financing", "residual")
    }
    clipped_rebalances = 0
    ill_conditioned_rebalances = 0
    condition_numbers: list[float] = []

    for step in range(steps):
        desired_stock, desired_options, clipped, ill_conditioned, conditions = _desired_positions(
            strategy, metrics[step], settings
        )
        clipped_rebalances += clipped
        ill_conditioned_rebalances += ill_conditioned
        condition_numbers.extend(conditions)

        stock_trade = desired_stock - stock_position
        option_trade = desired_options - option_position
        stock_notional = np.abs(stock_trade) * spots[step]
        option_notional = np.sum(np.abs(option_trade) * values[step, :, 1:], axis=1)
        step_cost = stock_notional * stock_cost_rate + option_notional * option_cost_rate
        cash -= stock_trade * spots[step]
        cash -= np.sum(option_trade * values[step, :, 1:], axis=1)
        cash -= step_cost
        transaction_cost += step_cost
        turnover += stock_notional + option_notional
        stock_position = desired_stock
        option_position = desired_options

        carry = cash * (cash_growth - 1.0) + stock_position * spots[step] * dividend_growth
        cash += carry
        attribution["financing"] += carry

        spot_change = spots[step + 1] - spots[step]
        volatility_change = np.sqrt(np.maximum(variances[step + 1], _VARIANCE_FLOOR)) - np.sqrt(
            np.maximum(variances[step], _VARIANCE_FLOOR)
        )
        target = metrics[step, :, 0]
        hedges = metrics[step, :, 1:]
        net_delta = (
            stock_position
            + np.sum(option_position * hedges[:, :, _DELTA], axis=1)
            - target[:, _DELTA]
        )
        net_gamma = np.sum(option_position * hedges[:, :, _GAMMA], axis=1) - target[:, _GAMMA]
        net_vega = np.sum(option_position * hedges[:, :, _VEGA], axis=1) - target[:, _VEGA]
        net_theta = np.sum(option_position * hedges[:, :, _THETA], axis=1) - target[:, _THETA]
        delta_attribution = net_delta * spot_change
        gamma_attribution = 0.5 * net_gamma * spot_change**2
        vega_attribution = net_vega * volatility_change
        theta_attribution = net_theta * time_step
        actual_risky_change = (
            stock_position * spot_change
            + np.sum(option_position * (values[step + 1, :, 1:] - values[step, :, 1:]), axis=1)
            - (values[step + 1, :, 0] - values[step, :, 0])
        )
        residual = actual_risky_change - (
            delta_attribution + gamma_attribution + vega_attribution + theta_attribution
        )
        attribution["delta"] += delta_attribution
        attribution["gamma"] += gamma_attribution
        attribution["vega"] += vega_attribution
        attribution["theta"] += theta_attribution
        attribution["residual"] += residual

    liquidation_notional = np.abs(stock_position) * spots[-1]
    liquidation_cost = liquidation_notional * stock_cost_rate
    transaction_cost += liquidation_cost
    turnover += liquidation_notional
    pnl = (
        cash
        + stock_position * spots[-1]
        + np.sum(option_position * values[-1, :, 1:], axis=1)
        - values[-1, :, 0]
        - liquidation_cost
    )
    attributed_total = (
        attribution["delta"]
        + attribution["gamma"]
        + attribution["vega"]
        + attribution["theta"]
        + attribution["financing"]
        - transaction_cost
        + attribution["residual"]
    )
    gap = pnl - attributed_total
    mean_attribution = HedgePnLAttribution(
        delta=float(np.mean(attribution["delta"])),
        gamma=float(np.mean(attribution["gamma"])),
        vega=float(np.mean(attribution["vega"])),
        theta=float(np.mean(attribution["theta"])),
        financing=float(np.mean(attribution["financing"])),
        transaction_cost=-float(np.mean(transaction_cost)),
        residual=float(np.mean(attribution["residual"])),
        total=float(np.mean(attributed_total)),
    )
    sample_deviation = float(np.std(pnl, ddof=1))
    conditions_array = np.asarray(condition_numbers, dtype=float)
    return HedgeStrategyResult(
        strategy=strategy,
        mean_pnl=float(np.mean(pnl)),
        pnl_standard_deviation=sample_deviation,
        pnl_standard_error=sample_deviation / math.sqrt(paths),
        pnl_q05=float(np.quantile(pnl, 0.05)),
        pnl_median=float(np.median(pnl)),
        pnl_q95=float(np.quantile(pnl, 0.95)),
        mean_transaction_cost=float(np.mean(transaction_cost)),
        mean_turnover=float(np.mean(turnover)),
        mean_attribution=mean_attribution,
        max_abs_attribution_gap=float(np.max(np.abs(gap))),
        clipped_rebalances=clipped_rebalances,
        ill_conditioned_rebalances=ill_conditioned_rebalances,
        mean_condition_number=(float(np.mean(conditions_array)) if conditions_array.size else None),
        max_condition_number=(float(np.max(conditions_array)) if conditions_array.size else None),
        pnl=tuple(float(value) for value in pnl),
    )


def run_heston_hedge_backtest(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    settings: HedgeBacktestSettings = _DEFAULT_SETTINGS,
    cos_settings: COSSettings = _DEFAULT_COS_SETTINGS,
    path_parameters: HestonParameters | None = None,
    path_spot_drift: float | None = None,
) -> HedgeBacktestResult:
    """Compare five self-financing hedge strategies on common Heston paths.

    The target is short one European option. Hedge options are calls with the
    same expiry. Vega is sensitivity to the current instantaneous volatility
    ``sqrt(v_t)`` while long-run variance remains fixed.
    """
    kind = OptionType.parse(option_type)
    _validate_contract(spot, strike, maturity, rate, dividend_yield)
    realized_parameters = parameters if path_parameters is None else path_parameters
    realized_drift = rate - dividend_yield if path_spot_drift is None else path_spot_drift
    if not math.isfinite(realized_drift):
        raise ValueError("path_spot_drift must be finite when supplied")
    path_simulation = simulate_heston_paths(
        spot,
        maturity,
        realized_parameters,
        rate,
        dividend_yield,
        settings=HestonPathSettings(
            paths=settings.paths,
            steps=settings.rebalancing_steps,
            seed=settings.seed,
            antithetic=settings.antithetic,
        ),
        spot_drift=realized_drift,
    )
    hedge_strikes = np.asarray(
        [strike * settings.hedge_strike_low, strike * settings.hedge_strike_high],
        dtype=float,
    )
    metrics, values = _precompute_metrics(
        kind,
        path_simulation.spots,
        path_simulation.variances,
        strike,
        hedge_strikes,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings,
        cos_settings,
    )
    strategies = tuple(
        _strategy_result(
            strategy,
            path_simulation.spots,
            path_simulation.variances,
            metrics,
            values,
            rate,
            dividend_yield,
            settings,
            maturity,
        )
        for strategy in HEDGE_STRATEGIES
    )
    return HedgeBacktestResult(
        target_option_type=kind.value,
        target_strike=strike,
        maturity=maturity,
        initial_option_price=float(values[0, 0, 0]),
        hedge_strikes=(float(hedge_strikes[0]), float(hedge_strikes[1])),
        settings=settings,
        pricing_parameters=parameters,
        path_parameters=realized_parameters,
        path_spot_drift=realized_drift,
        vega_variable="instantaneous volatility sqrt(v_t), with theta fixed",
        theta_convention="calendar-time decay -dV/dT per year",
        strategies=strategies,
    )
