"""Automatic refinement and consensus gating for Heston numerical engines."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from volforge.models.common import OptionType
from volforge.models.heston import COSSettings, HestonParameters, price
from volforge.numerics.monte_carlo import MonteCarloSettings, heston_monte_carlo
from volforge.numerics.pde import PDEGridSettings, heston_pde_price


class NumericalConvergenceError(RuntimeError):
    """Raised when a numerical result fails the convergence gate."""


@dataclass(frozen=True, slots=True)
class ConvergenceSettings:
    """Maximum refinement budget and acceptance tolerances.

    Each engine is evaluated at three automatically derived resolutions. The
    maximum fields define the third and most expensive level.
    """

    absolute_tolerance: float = 0.03
    relative_tolerance: float = 0.01
    max_cos_terms: int = 512
    max_cos_truncation_width: float = 20.0
    max_monte_carlo_paths: int = 40_000
    max_monte_carlo_steps: int = 128
    seed: int = 7
    antithetic: bool = True
    control_variate: bool = True
    max_pde_log_price_points: int = 121
    max_pde_variance_points: int = 61
    max_pde_time_steps: int = 180
    max_pde_log_price_half_width: float = 5.0
    max_pde_variance: float = 1.25

    def __post_init__(self) -> None:
        finite_values = (
            self.absolute_tolerance,
            self.relative_tolerance,
            self.max_cos_truncation_width,
            self.max_pde_log_price_half_width,
            self.max_pde_variance,
        )
        if any(not math.isfinite(value) for value in finite_values):
            raise ValueError("convergence settings must be finite")
        if self.absolute_tolerance < 0.0 or self.relative_tolerance < 0.0:
            raise ValueError("convergence tolerances must be non-negative")
        if self.absolute_tolerance == 0.0 and self.relative_tolerance == 0.0:
            raise ValueError("at least one convergence tolerance must be positive")
        if not isinstance(self.max_cos_terms, int) or self.max_cos_terms < 128:
            raise ValueError("max_cos_terms must be an integer of at least 128")
        if not 10.0 <= self.max_cos_truncation_width <= 30.0:
            raise ValueError("max_cos_truncation_width must be between 10 and 30")
        if not isinstance(self.max_monte_carlo_paths, int) or self.max_monte_carlo_paths < 4_000:
            raise ValueError("max_monte_carlo_paths must be an integer of at least 4000")
        if self.antithetic and self.max_monte_carlo_paths % 2:
            raise ValueError("antithetic convergence requires an even maximum path count")
        if not isinstance(self.max_monte_carlo_steps, int) or self.max_monte_carlo_steps < 16:
            raise ValueError("max_monte_carlo_steps must be an integer of at least 16")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("convergence seed must be a non-negative integer")
        if not isinstance(self.max_pde_log_price_points, int) or self.max_pde_log_price_points < 61:
            raise ValueError("max_pde_log_price_points must be an integer of at least 61")
        if not isinstance(self.max_pde_variance_points, int) or self.max_pde_variance_points < 31:
            raise ValueError("max_pde_variance_points must be an integer of at least 31")
        if not isinstance(self.max_pde_time_steps, int) or self.max_pde_time_steps < 60:
            raise ValueError("max_pde_time_steps must be an integer of at least 60")
        if self.max_pde_log_price_half_width <= 1.0:
            raise ValueError("max_pde_log_price_half_width must exceed one")
        if self.max_pde_variance <= 0.0:
            raise ValueError("max_pde_variance must be positive")

    def tolerance_for(self, value: float) -> float:
        """Return the scaled price tolerance at ``value``."""
        return self.absolute_tolerance + self.relative_tolerance * max(abs(value), 1.0)


@dataclass(frozen=True, slots=True)
class ConvergenceObservation:
    """One engine value at one automatically selected refinement level."""

    engine: str
    level: int
    configuration: str
    price: float
    change_from_previous: float | None
    uncertainty: float | None


@dataclass(frozen=True, slots=True)
class EngineConvergence:
    """Final convergence decision for one numerical engine."""

    engine: str
    converged: bool
    final_price: float
    estimated_error: float
    observations: tuple[ConvergenceObservation, ...]
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HestonConvergenceReport:
    """Three-engine convergence and consensus audit."""

    option_type: OptionType
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    parameters: HestonParameters
    settings: ConvergenceSettings
    downstream_eligible: bool
    reference_price: float
    consensus_price: float
    consensus_spread: float
    tolerance: float
    monte_carlo_reference_covered: bool
    monte_carlo_confidence_low: float
    monte_carlo_confidence_high: float
    engines: tuple[EngineConvergence, ...]
    failure_reasons: tuple[str, ...]

    @property
    def observations(self) -> tuple[ConvergenceObservation, ...]:
        """Return all refinement observations in engine order."""
        return tuple(observation for engine in self.engines for observation in engine.observations)


_DEFAULT_CONVERGENCE_SETTINGS = ConvergenceSettings()
_LEVELS = (1, 2, 3)


def _integer_level(
    maximum: int,
    ratio: float,
    minimum: int,
    *,
    even: bool = False,
    odd: bool = False,
) -> int:
    value = max(minimum, round(maximum * ratio))
    value = min(value, maximum)
    if even and value % 2:
        value = value + 1 if value < maximum else value - 1
    if odd and value % 2 == 0:
        value = value + 1 if value < maximum else value - 1
    return value


def _change(current: float, previous: float | None) -> float | None:
    return None if previous is None else abs(current - previous)


def _cos_convergence(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float,
    dividend_yield: float,
    settings: ConvergenceSettings,
) -> tuple[EngineConvergence, float]:
    observations: list[ConvergenceObservation] = []
    previous: float | None = None
    term_ratios = (0.25, 0.5, 1.0)
    width_ratios = (0.6, 0.8, 1.0)
    for level, term_ratio, width_ratio in zip(_LEVELS, term_ratios, width_ratios, strict=True):
        terms = _integer_level(settings.max_cos_terms, term_ratio, 32)
        width = max(5.0, settings.max_cos_truncation_width * width_ratio)
        model_price = price(
            option_type,
            spot,
            strike,
            maturity,
            parameters,
            rate,
            dividend_yield,
            settings=COSSettings(terms=terms, truncation_width=width),
        )
        observations.append(
            ConvergenceObservation(
                engine="COS",
                level=level,
                configuration=f"terms={terms}, width={width:.3g}",
                price=model_price,
                change_from_previous=_change(model_price, previous),
                uncertainty=None,
            )
        )
        previous = model_price

    final_price = observations[-1].price
    estimated_error = observations[-1].change_from_previous or 0.0
    tolerance = settings.tolerance_for(final_price)
    reasons = (
        (f"COS final refinement change {estimated_error:.6g} exceeds tolerance {tolerance:.6g}",)
        if estimated_error > tolerance
        else ()
    )
    return (
        EngineConvergence(
            engine="COS",
            converged=not reasons,
            final_price=final_price,
            estimated_error=estimated_error,
            observations=tuple(observations),
            failure_reasons=reasons,
        ),
        final_price,
    )


def _monte_carlo_convergence(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float,
    dividend_yield: float,
    settings: ConvergenceSettings,
) -> tuple[EngineConvergence, float, float, bool]:
    observations: list[ConvergenceObservation] = []
    previous: float | None = None
    final_result = None
    ratios = (0.25, 0.5, 1.0)
    for level, ratio in zip(_LEVELS, ratios, strict=True):
        paths = _integer_level(
            settings.max_monte_carlo_paths,
            ratio,
            2,
            even=settings.antithetic,
        )
        steps = _integer_level(settings.max_monte_carlo_steps, ratio, 1)
        final_result = heston_monte_carlo(
            option_type,
            spot,
            strike,
            maturity,
            parameters,
            rate,
            dividend_yield,
            settings=MonteCarloSettings(
                paths=paths,
                steps=steps,
                seed=settings.seed,
                antithetic=settings.antithetic,
                control_variate=settings.control_variate,
            ),
        )
        half_width = 1.959963984540054 * final_result.standard_error
        observations.append(
            ConvergenceObservation(
                engine="QE Monte Carlo",
                level=level,
                configuration=f"paths={paths}, steps={steps}, seed={settings.seed}",
                price=final_result.price,
                change_from_previous=_change(final_result.price, previous),
                uncertainty=half_width,
            )
        )
        previous = final_result.price

    if final_result is None:  # pragma: no cover - fixed refinement ladder
        raise RuntimeError("Monte Carlo refinement ladder produced no result")
    final_observation = observations[-1]
    estimated_error = max(
        final_observation.change_from_previous or 0.0,
        final_observation.uncertainty or 0.0,
    )
    tolerance = settings.tolerance_for(final_result.price)
    reasons: list[str] = []
    if estimated_error > tolerance:
        reasons.append(
            f"Monte Carlo error estimate {estimated_error:.6g} exceeds tolerance {tolerance:.6g}"
        )
    if not final_result.estimate_within_bounds:
        reasons.append("Monte Carlo point estimate violates European price bounds")
    return (
        EngineConvergence(
            engine="QE Monte Carlo",
            converged=not reasons,
            final_price=final_result.price,
            estimated_error=estimated_error,
            observations=tuple(observations),
            failure_reasons=tuple(reasons),
        ),
        final_result.confidence_low,
        final_result.confidence_high,
        final_result.estimate_within_bounds,
    )


def _pde_convergence(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float,
    dividend_yield: float,
    settings: ConvergenceSettings,
) -> EngineConvergence:
    if settings.max_pde_variance <= parameters.initial_variance:
        raise ValueError("max_pde_variance must exceed the Heston initial variance")
    observations: list[ConvergenceObservation] = []
    previous: float | None = None
    point_ratios = (0.5, 0.75, 1.0)
    domain_ratios = (0.6, 0.8, 1.0)
    for level, point_ratio, domain_ratio in zip(_LEVELS, point_ratios, domain_ratios, strict=True):
        log_points = _integer_level(settings.max_pde_log_price_points, point_ratio, 21, odd=True)
        variance_points = _integer_level(
            settings.max_pde_variance_points, point_ratio, 11, odd=True
        )
        time_steps = _integer_level(settings.max_pde_time_steps, point_ratio, 1)
        half_width = max(0.51, settings.max_pde_log_price_half_width * domain_ratio)
        variance_max = max(
            settings.max_pde_variance * domain_ratio,
            parameters.initial_variance * (1.0 + 1e-6),
        )
        pde_result = heston_pde_price(
            option_type,
            spot,
            strike,
            maturity,
            parameters,
            rate,
            dividend_yield,
            settings=PDEGridSettings(
                log_price_points=log_points,
                variance_points=variance_points,
                time_steps=time_steps,
                log_price_half_width=half_width,
                variance_max=variance_max,
            ),
        )
        observations.append(
            ConvergenceObservation(
                engine="Finite-difference PDE",
                level=level,
                configuration=(
                    f"grid={log_points}x{variance_points}x{time_steps}, "
                    f"width={half_width:.3g}, vmax={variance_max:.3g}"
                ),
                price=pde_result.price,
                change_from_previous=_change(pde_result.price, previous),
                uncertainty=None,
            )
        )
        previous = pde_result.price

    final_price = observations[-1].price
    estimated_error = observations[-1].change_from_previous or 0.0
    tolerance = settings.tolerance_for(final_price)
    reasons = (
        (f"PDE final refinement change {estimated_error:.6g} exceeds tolerance {tolerance:.6g}",)
        if estimated_error > tolerance
        else ()
    )
    return EngineConvergence(
        engine="Finite-difference PDE",
        converged=not reasons,
        final_price=final_price,
        estimated_error=estimated_error,
        observations=tuple(observations),
        failure_reasons=reasons,
    )


def assess_heston_convergence(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    settings: ConvergenceSettings = _DEFAULT_CONVERGENCE_SETTINGS,
) -> HestonConvergenceReport:
    """Run three refinement ladders and return a downstream eligibility decision."""
    kind = OptionType.parse(option_type)
    cos_report, reference_price = _cos_convergence(
        kind, spot, strike, maturity, parameters, rate, dividend_yield, settings
    )
    monte_carlo_report, confidence_low, confidence_high, _ = _monte_carlo_convergence(
        kind, spot, strike, maturity, parameters, rate, dividend_yield, settings
    )
    pde_report = _pde_convergence(
        kind, spot, strike, maturity, parameters, rate, dividend_yield, settings
    )
    engines = (cos_report, monte_carlo_report, pde_report)
    final_prices = np.asarray([engine.final_price for engine in engines], dtype=float)
    consensus_price = float(np.median(final_prices))
    consensus_spread = float(np.max(final_prices) - np.min(final_prices))
    tolerance = settings.tolerance_for(reference_price)
    monte_carlo_reference_covered = confidence_low <= reference_price <= confidence_high

    failure_reasons = [reason for engine in engines for reason in engine.failure_reasons]
    if consensus_spread > tolerance:
        failure_reasons.append(
            f"cross-engine spread {consensus_spread:.6g} exceeds tolerance {tolerance:.6g}"
        )
    if not monte_carlo_reference_covered:
        failure_reasons.append("COS reference lies outside the Monte Carlo 95% confidence interval")
    return HestonConvergenceReport(
        option_type=kind,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend_yield,
        parameters=parameters,
        settings=settings,
        downstream_eligible=not failure_reasons,
        reference_price=reference_price,
        consensus_price=consensus_price,
        consensus_spread=consensus_spread,
        tolerance=tolerance,
        monte_carlo_reference_covered=monte_carlo_reference_covered,
        monte_carlo_confidence_low=confidence_low,
        monte_carlo_confidence_high=confidence_high,
        engines=engines,
        failure_reasons=tuple(failure_reasons),
    )


def require_numerical_convergence(
    report: HestonConvergenceReport,
) -> HestonConvergenceReport:
    """Return ``report`` only when all refinement and consensus checks pass."""
    if report.downstream_eligible:
        return report
    reasons = report.failure_reasons or ("numerical convergence failed without a reason",)
    raise NumericalConvergenceError("price is blocked from downstream use: " + "; ".join(reasons))


def convergence_report_json(report: HestonConvergenceReport) -> bytes:
    """Serialize a convergence report to deterministic UTF-8 JSON."""
    payload = json.dumps(asdict(report), indent=2, sort_keys=True, allow_nan=False)
    return f"{payload}\n".encode()
