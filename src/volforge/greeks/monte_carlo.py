"""Direct and common-random-number Heston Monte Carlo Greeks."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from volforge.models.common import OptionType
from volforge.models.heston import HestonParameters
from volforge.numerics.monte_carlo import HestonPathSettings, simulate_heston_paths


@dataclass(frozen=True, slots=True)
class MonteCarloGreekSettings:
    """Simulation budget and CRN bump controls."""

    paths: int = 20_000
    steps: int = 64
    seed: int = 7
    antithetic: bool = True
    psi_threshold: float = 1.5
    initial_volatility_relative_bump: float = 1e-2
    maturity_relative_bump: float = 1e-2

    def __post_init__(self) -> None:
        HestonPathSettings(self.paths, self.steps, self.seed, self.antithetic, self.psi_threshold)
        if self.antithetic and self.paths < 4:
            raise ValueError("Monte Carlo Greeks require at least four antithetic paths")
        if (
            not math.isfinite(self.initial_volatility_relative_bump)
            or not 0.0 < self.initial_volatility_relative_bump <= 0.1
        ):
            raise ValueError("initial-volatility bump must lie in (0, 0.1]")
        if (
            not math.isfinite(self.maturity_relative_bump)
            or not 0.0 < self.maturity_relative_bump <= 0.1
        ):
            raise ValueError("maturity bump must lie in (0, 0.1]")


@dataclass(frozen=True, slots=True)
class MonteCarloGreekEstimate:
    """Sampling estimate with a normal-approximation interval."""

    value: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    method: str


@dataclass(frozen=True, slots=True)
class MonteCarloGreekReport:
    """A complete seeded Monte Carlo price-and-Greeks run."""

    price: MonteCarloGreekEstimate
    delta: MonteCarloGreekEstimate
    gamma: MonteCarloGreekEstimate
    vega: MonteCarloGreekEstimate
    theta: MonteCarloGreekEstimate
    rho: MonteCarloGreekEstimate
    delta_likelihood_ratio: MonteCarloGreekEstimate
    paths: int
    steps: int
    seed: int
    antithetic: bool
    vega_variable: str
    theta_convention: str


_DEFAULT_SETTINGS = MonteCarloGreekSettings()


def _payoff(kind: OptionType, terminal_spot: np.ndarray, strike: float) -> np.ndarray:
    if kind is OptionType.CALL:
        return np.maximum(terminal_spot - strike, 0.0)
    return np.maximum(strike - terminal_spot, 0.0)


def _independent(values: np.ndarray, antithetic: bool) -> np.ndarray:
    if not antithetic:
        return values
    half = len(values) // 2
    return 0.5 * (values[:half] + values[half:])


def _estimate(observations: np.ndarray, antithetic: bool, method: str) -> MonteCarloGreekEstimate:
    independent = _independent(observations, antithetic)
    value = float(independent.mean())
    standard_error = float(independent.std(ddof=1) / math.sqrt(len(independent)))
    if not math.isfinite(value) or not math.isfinite(standard_error):
        raise ArithmeticError("Monte Carlo Greek produced a non-finite estimate")
    half_width = 1.959963984540054 * standard_error
    return MonteCarloGreekEstimate(
        value, standard_error, value - half_width, value + half_width, method
    )


def _path_settings(settings: MonteCarloGreekSettings) -> HestonPathSettings:
    return HestonPathSettings(
        settings.paths,
        settings.steps,
        settings.seed,
        settings.antithetic,
        settings.psi_threshold,
    )


def _discounted_payoffs(
    kind: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float,
    dividend_yield: float,
    settings: MonteCarloGreekSettings,
) -> np.ndarray:
    simulation = simulate_heston_paths(
        spot,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=_path_settings(settings),
    )
    return math.exp(-rate * maturity) * _payoff(kind, simulation.spots[-1], strike)


def heston_monte_carlo_greeks(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    settings: MonteCarloGreekSettings = _DEFAULT_SETTINGS,
) -> MonteCarloGreekReport:
    """Estimate Heston Greeks with direct, LR, and paired-CRN estimators."""
    kind = OptionType.parse(option_type)
    if not math.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be positive and finite")
    if not math.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be positive and finite")
    if maturity <= 0.0 or not math.isfinite(maturity):
        raise ValueError("Heston Greeks require positive finite maturity")
    simulation = simulate_heston_paths(
        spot,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=_path_settings(settings),
    )
    terminal_spot = simulation.spots[-1]
    discount = math.exp(-rate * maturity)
    payoff = _payoff(kind, terminal_spot, strike)
    discounted_payoff = discount * payoff
    in_money = terminal_spot > strike if kind is OptionType.CALL else terminal_spot < strike
    payoff_slope = 1.0 if kind is OptionType.CALL else -1.0
    pathwise_delta = discount * payoff_slope * terminal_spot / spot * in_money
    pathwise_rho = (
        maturity * discount * strike * (1.0 if kind is OptionType.CALL else -1.0) * in_money
    )

    conditional_variance = simulation.conditional_log_variance
    if np.any(conditional_variance <= np.finfo(float).eps):
        raise ArithmeticError("likelihood-ratio variance is numerically zero")
    score_log_spot = simulation.conditional_log_shock / conditional_variance
    likelihood_delta = discounted_payoff * score_log_spot / spot
    likelihood_gamma = (
        discounted_payoff
        * (score_log_spot**2 - score_log_spot - 1.0 / conditional_variance)
        / spot**2
    )

    initial_volatility = math.sqrt(parameters.initial_variance)
    long_run_volatility = math.sqrt(parameters.theta)
    volatility_bump = min(
        max(initial_volatility * settings.initial_volatility_relative_bump, 1e-5),
        0.2 * min(initial_volatility, long_run_volatility),
    )
    plus_parameters = replace(
        parameters,
        initial_variance=(initial_volatility + volatility_bump) ** 2,
        theta=(long_run_volatility + volatility_bump) ** 2,
    )
    minus_parameters = replace(
        parameters,
        initial_variance=(initial_volatility - volatility_bump) ** 2,
        theta=(long_run_volatility - volatility_bump) ** 2,
    )
    vega_plus = _discounted_payoffs(
        kind,
        spot,
        strike,
        maturity,
        plus_parameters,
        rate,
        dividend_yield,
        settings,
    )
    vega_minus = _discounted_payoffs(
        kind,
        spot,
        strike,
        maturity,
        minus_parameters,
        rate,
        dividend_yield,
        settings,
    )
    vega_observations = (vega_plus - vega_minus) / (2.0 * volatility_bump)

    maturity_bump = min(max(maturity * settings.maturity_relative_bump, 1e-5), 0.2 * maturity)
    theta_plus = _discounted_payoffs(
        kind,
        spot,
        strike,
        maturity + maturity_bump,
        parameters,
        rate,
        dividend_yield,
        settings,
    )
    theta_minus = _discounted_payoffs(
        kind,
        spot,
        strike,
        maturity - maturity_bump,
        parameters,
        rate,
        dividend_yield,
        settings,
    )
    theta_observations = -(theta_plus - theta_minus) / (2.0 * maturity_bump)
    method_crn = "paired common-random-number central bump"
    return MonteCarloGreekReport(
        price=_estimate(discounted_payoff, settings.antithetic, "discounted payoff"),
        delta=_estimate(pathwise_delta, settings.antithetic, "pathwise"),
        gamma=_estimate(likelihood_gamma, settings.antithetic, "conditional likelihood ratio"),
        vega=_estimate(vega_observations, settings.antithetic, method_crn),
        theta=_estimate(theta_observations, settings.antithetic, method_crn),
        rho=_estimate(pathwise_rho, settings.antithetic, "pathwise"),
        delta_likelihood_ratio=_estimate(
            likelihood_delta, settings.antithetic, "conditional likelihood ratio"
        ),
        paths=settings.paths,
        steps=settings.steps,
        seed=settings.seed,
        antithetic=settings.antithetic,
        vega_variable="parallel shift to sqrt(v0) and sqrt(theta)",
        theta_convention="calendar decay -dV/dT per year",
    )
