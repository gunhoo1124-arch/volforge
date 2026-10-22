"""Deterministic Heston Monte Carlo with Andersen QE-M variance stepping."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from volforge.models.common import OptionType, discounted_price_bounds
from volforge.models.heston import HestonParameters

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MonteCarloSettings:
    """Simulation controls and variance-reduction choices."""

    paths: int = 100_000
    steps: int = 252
    seed: int = 7
    antithetic: bool = True
    control_variate: bool = True
    psi_threshold: float = 1.5

    def __post_init__(self) -> None:
        if not isinstance(self.paths, int) or self.paths < 2:
            raise ValueError("Monte Carlo paths must be an integer of at least two")
        if self.antithetic and self.paths % 2:
            raise ValueError("antithetic Monte Carlo requires an even path count")
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("Monte Carlo steps must be a positive integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Monte Carlo seed must be a non-negative integer")
        if not math.isfinite(self.psi_threshold) or not 1.0 < self.psi_threshold < 2.0:
            raise ValueError("QE psi_threshold must lie strictly between one and two")


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """A simulated price with sampling uncertainty and run diagnostics."""

    price: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    lower_bound: float
    upper_bound: float
    estimate_within_bounds: bool
    paths: int
    steps: int
    seed: int
    antithetic: bool
    control_variate: bool
    control_beta: float
    discounted_underlying_mean: float


@dataclass(frozen=True, slots=True)
class HestonPathSettings:
    """Controls for a reproducible full-path Heston simulation."""

    paths: int = 20_000
    steps: int = 64
    seed: int = 7
    antithetic: bool = True
    psi_threshold: float = 1.5

    def __post_init__(self) -> None:
        if not isinstance(self.paths, int) or self.paths < 2:
            raise ValueError("Heston path count must be an integer of at least two")
        if self.antithetic and self.paths % 2:
            raise ValueError("antithetic Heston paths require an even path count")
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("Heston path steps must be a positive integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Heston path seed must be a non-negative integer")
        if not math.isfinite(self.psi_threshold) or not 1.0 < self.psi_threshold < 2.0:
            raise ValueError("QE psi_threshold must lie strictly between one and two")


@dataclass(frozen=True, slots=True)
class HestonPathSimulation:
    """Spot and variance paths, including the initial state."""

    spots: FloatArray
    variances: FloatArray
    time_step: float
    seed: int
    antithetic: bool
    conditional_log_variance: FloatArray
    conditional_log_shock: FloatArray


_DEFAULT_MONTE_CARLO_SETTINGS = MonteCarloSettings()
_DEFAULT_PATH_SETTINGS = HestonPathSettings()


def _validate_contract(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
) -> None:
    values = {
        "spot": spot,
        "strike": strike,
        "maturity": maturity,
        "rate": rate,
        "dividend_yield": dividend_yield,
    }
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) for value in values.values()
    ):
        raise ValueError("contract inputs must be finite numbers")
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")


def _random_draws(
    generator: np.random.Generator,
    settings: MonteCarloSettings | HestonPathSettings,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    draw_count = settings.paths // 2 if settings.antithetic else settings.paths
    variance_normal = generator.standard_normal(draw_count)
    spot_normal = generator.standard_normal(draw_count)
    uniform = generator.random(draw_count)
    if not settings.antithetic:
        return variance_normal, spot_normal, uniform
    return (
        np.concatenate((variance_normal, -variance_normal)),
        np.concatenate((spot_normal, -spot_normal)),
        np.concatenate((uniform, 1.0 - uniform)),
    )


def _qe_variance_step(
    variance: FloatArray,
    variance_normal: FloatArray,
    uniform: FloatArray,
    parameters: HestonParameters,
    time_step: float,
    psi_threshold: float,
) -> tuple[FloatArray, FloatArray]:
    kappa = parameters.kappa
    theta = parameters.theta
    sigma = parameters.vol_of_vol
    decay = math.exp(-kappa * time_step)
    mean = theta + (variance - theta) * decay
    conditional_variance = variance * sigma**2 * decay * (
        1.0 - decay
    ) / kappa + theta * sigma**2 * (1.0 - decay) ** 2 / (2.0 * kappa)
    psi = np.maximum(conditional_variance / np.maximum(mean**2, np.finfo(float).tiny), 0.0)
    low = psi <= psi_threshold
    next_variance = np.empty_like(variance)
    if np.any(low):
        psi_low = np.maximum(psi[low], np.finfo(float).eps)
        two_over_psi = 2.0 / psi_low
        b_squared = two_over_psi - 1.0 + np.sqrt(two_over_psi * (two_over_psi - 1.0))
        a = mean[low] / (1.0 + b_squared)
        next_variance[low] = a * (np.sqrt(b_squared) + variance_normal[low]) ** 2

    high = ~low
    if np.any(high):
        psi_high = psi[high]
        probability_zero = (psi_high - 1.0) / (psi_high + 1.0)
        beta = (1.0 - probability_zero) / mean[high]
        safe_uniform = np.clip(uniform[high], np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        next_variance[high] = np.where(
            safe_uniform <= probability_zero,
            0.0,
            np.log((1.0 - probability_zero) / (1.0 - safe_uniform)) / beta,
        )
    return next_variance, psi


def _martingale_correction(
    variance: FloatArray,
    psi: FloatArray,
    parameters: HestonParameters,
    time_step: float,
    psi_threshold: float,
    a2: float,
    k1_plus_half_k3: float,
) -> FloatArray:
    kappa = parameters.kappa
    theta = parameters.theta
    decay = math.exp(-kappa * time_step)
    mean = theta + (variance - theta) * decay
    correction = np.empty_like(variance)
    low = psi <= psi_threshold

    if np.any(low):
        psi_low = np.maximum(psi[low], np.finfo(float).eps)
        two_over_psi = 2.0 / psi_low
        b_squared = two_over_psi - 1.0 + np.sqrt(two_over_psi * (two_over_psi - 1.0))
        a = mean[low] / (1.0 + b_squared)
        denominator = 1.0 - 2.0 * a2 * a
        if np.any(denominator <= 0.0):
            raise ArithmeticError("QE martingale correction is undefined for these settings")
        log_mgf = a2 * a * b_squared / denominator - 0.5 * np.log(denominator)
        correction[low] = -log_mgf - k1_plus_half_k3 * variance[low]

    high = ~low
    if np.any(high):
        psi_high = psi[high]
        probability_zero = (psi_high - 1.0) / (psi_high + 1.0)
        beta = (1.0 - probability_zero) / mean[high]
        if np.any(beta <= a2):
            raise ArithmeticError("QE martingale correction is undefined for these settings")
        mgf = probability_zero + beta * (1.0 - probability_zero) / (beta - a2)
        correction[high] = -np.log(mgf) - k1_plus_half_k3 * variance[high]
    return correction


def simulate_heston_paths(
    spot: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    settings: HestonPathSettings = _DEFAULT_PATH_SETTINGS,
    spot_drift: float | None = None,
) -> HestonPathSimulation:
    """Simulate complete Heston paths with Andersen QE-M transitions.

    ``spot_drift`` is the ex-dividend drift coefficient in ``dS/S``. Its
    default is the risk-neutral value ``rate - dividend_yield``; supplying it
    permits physical-measure or stressed path dynamics without changing the
    discounting inputs used elsewhere.
    """
    _validate_contract(spot, spot, maturity, rate, dividend_yield)
    if spot_drift is not None and not math.isfinite(spot_drift):
        raise ValueError("spot_drift must be finite when supplied")
    time_step = maturity / settings.steps if maturity else 0.0
    spots = np.full((settings.steps + 1, settings.paths), spot, dtype=float)
    variances = np.full(
        (settings.steps + 1, settings.paths), parameters.initial_variance, dtype=float
    )
    conditional_log_variance = np.zeros(settings.paths, dtype=float)
    conditional_log_shock = np.zeros(settings.paths, dtype=float)
    if maturity == 0.0:
        return HestonPathSimulation(
            spots,
            variances,
            time_step,
            settings.seed,
            settings.antithetic,
            conditional_log_variance,
            conditional_log_shock,
        )

    rho = parameters.rho
    sigma = parameters.vol_of_vol
    gamma1 = gamma2 = 0.5
    k1 = gamma1 * time_step * (parameters.kappa * rho / sigma - 0.5) - rho / sigma
    k2 = gamma2 * time_step * (parameters.kappa * rho / sigma - 0.5) + rho / sigma
    k3 = gamma1 * time_step * (1.0 - rho**2)
    k4 = gamma2 * time_step * (1.0 - rho**2)
    a2 = k2 + 0.5 * k4
    k1_plus_half_k3 = k1 + 0.5 * k3
    drift = (rate - dividend_yield if spot_drift is None else float(spot_drift)) * time_step
    generator = np.random.default_rng(settings.seed)
    log_spot = np.full(settings.paths, math.log(spot), dtype=float)
    variance = np.full(settings.paths, parameters.initial_variance, dtype=float)

    for step in range(1, settings.steps + 1):
        variance_normal, spot_normal, uniform = _random_draws(generator, settings)
        next_variance, psi = _qe_variance_step(
            variance,
            variance_normal,
            uniform,
            parameters,
            time_step,
            settings.psi_threshold,
        )
        correction = _martingale_correction(
            variance,
            psi,
            parameters,
            time_step,
            settings.psi_threshold,
            a2,
            k1_plus_half_k3,
        )
        conditional_variance = np.maximum(k3 * variance + k4 * next_variance, 0.0)
        conditional_log_variance += conditional_variance
        conditional_log_shock += np.sqrt(conditional_variance) * spot_normal
        log_spot += (
            drift
            + correction
            + k1 * variance
            + k2 * next_variance
            + np.sqrt(conditional_variance) * spot_normal
        )
        variance = next_variance
        spots[step] = np.exp(log_spot)
        variances[step] = variance
    return HestonPathSimulation(
        spots,
        variances,
        time_step,
        settings.seed,
        settings.antithetic,
        conditional_log_variance,
        conditional_log_shock,
    )


def heston_monte_carlo(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    settings: MonteCarloSettings = _DEFAULT_MONTE_CARLO_SETTINGS,
) -> MonteCarloResult:
    """Price a European option with seeded Andersen QE-M simulation.

    Antithetic sampling pairs Gaussian draws and uniforms. The optional
    control variate uses the discounted terminal underlying, whose expectation
    is known under the risk-neutral measure.
    """
    kind = OptionType.parse(option_type)
    _validate_contract(spot, strike, maturity, rate, dividend_yield)
    if maturity == 0.0:
        payoff = max(spot - strike, 0.0) if kind is OptionType.CALL else max(strike - spot, 0.0)
        return MonteCarloResult(
            price=payoff,
            standard_error=0.0,
            confidence_low=payoff,
            confidence_high=payoff,
            lower_bound=payoff,
            upper_bound=spot if kind is OptionType.CALL else strike,
            estimate_within_bounds=True,
            paths=settings.paths,
            steps=settings.steps,
            seed=settings.seed,
            antithetic=settings.antithetic,
            control_variate=settings.control_variate,
            control_beta=0.0,
            discounted_underlying_mean=spot,
        )

    time_step = maturity / settings.steps
    rho = parameters.rho
    sigma = parameters.vol_of_vol
    gamma1 = gamma2 = 0.5
    k1 = gamma1 * time_step * (parameters.kappa * rho / sigma - 0.5) - rho / sigma
    k2 = gamma2 * time_step * (parameters.kappa * rho / sigma - 0.5) + rho / sigma
    k3 = gamma1 * time_step * (1.0 - rho**2)
    k4 = gamma2 * time_step * (1.0 - rho**2)
    a2 = k2 + 0.5 * k4
    k1_plus_half_k3 = k1 + 0.5 * k3

    generator = np.random.default_rng(settings.seed)
    log_spot = np.full(settings.paths, math.log(spot), dtype=float)
    variance = np.full(settings.paths, parameters.initial_variance, dtype=float)
    drift = (rate - dividend_yield) * time_step
    for _ in range(settings.steps):
        variance_normal, spot_normal, uniform = _random_draws(generator, settings)
        next_variance, psi = _qe_variance_step(
            variance,
            variance_normal,
            uniform,
            parameters,
            time_step,
            settings.psi_threshold,
        )
        correction = _martingale_correction(
            variance,
            psi,
            parameters,
            time_step,
            settings.psi_threshold,
            a2,
            k1_plus_half_k3,
        )
        conditional_variance = np.maximum(k3 * variance + k4 * next_variance, 0.0)
        log_spot += (
            drift
            + correction
            + k1 * variance
            + k2 * next_variance
            + np.sqrt(conditional_variance) * spot_normal
        )
        variance = next_variance

    terminal_spot = np.exp(log_spot)
    discount = math.exp(-rate * maturity)
    if kind is OptionType.CALL:
        observations = discount * np.maximum(terminal_spot - strike, 0.0)
    else:
        observations = discount * np.maximum(strike - terminal_spot, 0.0)
    discounted_underlying = discount * terminal_spot
    control_beta = 0.0
    if settings.control_variate:
        centered_control = discounted_underlying - discounted_underlying.mean()
        control_variance = float(np.dot(centered_control, centered_control))
        if control_variance > np.finfo(float).eps:
            centered_payoff = observations - observations.mean()
            control_beta = float(np.dot(centered_payoff, centered_control) / control_variance)
            observations = observations - control_beta * (
                discounted_underlying - spot * math.exp(-dividend_yield * maturity)
            )

    if settings.antithetic:
        half = settings.paths // 2
        independent_observations = 0.5 * (observations[:half] + observations[half:])
    else:
        independent_observations = observations
    model_price = float(independent_observations.mean())
    standard_error = float(
        independent_observations.std(ddof=1) / math.sqrt(len(independent_observations))
    )
    lower_bound, upper_bound = discounted_price_bounds(
        kind, spot, strike, maturity, rate, dividend_yield
    )
    if not math.isfinite(model_price) or not math.isfinite(standard_error):
        raise ArithmeticError("Monte Carlo produced a non-finite estimate")
    half_width = 1.959963984540054 * standard_error
    return MonteCarloResult(
        price=model_price,
        standard_error=standard_error,
        confidence_low=model_price - half_width,
        confidence_high=model_price + half_width,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        estimate_within_bounds=lower_bound <= model_price <= upper_bound,
        paths=settings.paths,
        steps=settings.steps,
        seed=settings.seed,
        antithetic=settings.antithetic,
        control_variate=settings.control_variate,
        control_beta=control_beta,
        discounted_underlying_mean=float(discounted_underlying.mean()),
    )
