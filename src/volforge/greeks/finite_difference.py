"""High-order bump-and-revalue Heston Greeks from COS or PDE prices."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

from volforge.models.common import OptionType
from volforge.models.heston import COSSettings, HestonParameters, price
from volforge.numerics.pde import PDEGridSettings, heston_pde_price


@dataclass(frozen=True, slots=True)
class FiniteDifferenceSettings:
    """Relative bump sizes for deterministic Greek estimates."""

    spot_relative_bump: float = 1e-3
    initial_volatility_relative_bump: float = 1e-3
    maturity_relative_bump: float = 1e-3
    rate_bump: float = 1e-4

    def __post_init__(self) -> None:
        values = (
            self.spot_relative_bump,
            self.initial_volatility_relative_bump,
            self.maturity_relative_bump,
            self.rate_bump,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("finite-difference bumps must be positive finite numbers")
        if any(value > 0.1 for value in values):
            raise ValueError("finite-difference bumps must not exceed 0.1")


@dataclass(frozen=True, slots=True)
class DeterministicGreekEstimate:
    """Five-point value with a same-bump three-point discrepancy."""

    value: float
    coarse_value: float
    estimated_error: float
    bump: float
    method: str


@dataclass(frozen=True, slots=True)
class HestonGreekReport:
    """One complete deterministic Heston Greek run."""

    engine: str
    price: float
    delta: DeterministicGreekEstimate
    gamma: DeterministicGreekEstimate
    vega: DeterministicGreekEstimate
    theta: DeterministicGreekEstimate
    rho: DeterministicGreekEstimate
    price_evaluations: int
    vega_variable: str
    theta_convention: str


_DEFAULT_SETTINGS = FiniteDifferenceSettings()
_DEFAULT_COS_SETTINGS = COSSettings()
_DEFAULT_PDE_SETTINGS = PDEGridSettings()


def _first_derivative(
    function: Callable[[float], float], center: float, bump: float
) -> tuple[float, float]:
    minus_two = function(center - 2.0 * bump)
    minus_one = function(center - bump)
    plus_one = function(center + bump)
    plus_two = function(center + 2.0 * bump)
    fine = (minus_two - 8.0 * minus_one + 8.0 * plus_one - plus_two) / (12.0 * bump)
    coarse = (plus_one - minus_one) / (2.0 * bump)
    return fine, coarse


def _second_derivative(
    function: Callable[[float], float], center: float, bump: float, center_value: float
) -> tuple[float, float]:
    minus_two = function(center - 2.0 * bump)
    minus_one = function(center - bump)
    plus_one = function(center + bump)
    plus_two = function(center + 2.0 * bump)
    fine = (-plus_two + 16.0 * plus_one - 30.0 * center_value + 16.0 * minus_one - minus_two) / (
        12.0 * bump**2
    )
    coarse = (plus_one - 2.0 * center_value + minus_one) / bump**2
    return fine, coarse


def _estimate(value: float, coarse: float, bump: float, method: str) -> DeterministicGreekEstimate:
    if not math.isfinite(value) or not math.isfinite(coarse):
        raise ArithmeticError("finite-difference Greek is non-finite")
    return DeterministicGreekEstimate(value, coarse, abs(value - coarse), bump, method)


def heston_finite_difference_greeks(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    engine: str = "cos",
    settings: FiniteDifferenceSettings = _DEFAULT_SETTINGS,
    cos_settings: COSSettings = _DEFAULT_COS_SETTINGS,
    pde_settings: PDEGridSettings = _DEFAULT_PDE_SETTINGS,
) -> HestonGreekReport:
    """Estimate delta, gamma, vega, theta, and rho with five-point stencils.

    Vega applies a parallel shift to ``sqrt(v0)`` and ``sqrt(theta)``;
    theta is calendar decay ``-dV/dT``.
    """
    kind = OptionType.parse(option_type)
    if maturity <= 0.0 or not math.isfinite(maturity):
        raise ValueError("Heston Greeks require positive finite maturity")
    if engine not in {"cos", "pde"}:
        raise ValueError("Greek engine must be 'cos' or 'pde'")

    def reprice(
        bumped_spot: float = spot,
        bumped_maturity: float = maturity,
        bumped_parameters: HestonParameters = parameters,
        bumped_rate: float = rate,
    ) -> float:
        if engine == "cos":
            return price(
                kind,
                bumped_spot,
                strike,
                bumped_maturity,
                bumped_parameters,
                bumped_rate,
                dividend_yield,
                settings=cos_settings,
            )
        return heston_pde_price(
            kind,
            bumped_spot,
            strike,
            bumped_maturity,
            bumped_parameters,
            bumped_rate,
            dividend_yield,
            settings=pde_settings,
            grid_center_spot=spot,
            interpolation="cubic",
        ).price

    model_price = reprice()
    spot_bump = min(max(spot * settings.spot_relative_bump, 1e-5), 0.2 * spot)
    delta_value, delta_coarse = _first_derivative(
        lambda bumped: reprice(bumped_spot=bumped), spot, spot_bump
    )
    gamma_value, gamma_coarse = _second_derivative(
        lambda bumped: reprice(bumped_spot=bumped), spot, spot_bump, model_price
    )

    initial_volatility = math.sqrt(parameters.initial_variance)
    long_run_volatility = math.sqrt(parameters.theta)
    volatility_bump = min(
        max(initial_volatility * settings.initial_volatility_relative_bump, 1e-5),
        0.2 * min(initial_volatility, long_run_volatility),
    )

    def volatility_price(volatility: float) -> float:
        shift = volatility - initial_volatility
        return reprice(
            bumped_parameters=replace(
                parameters,
                initial_variance=volatility**2,
                theta=(long_run_volatility + shift) ** 2,
            )
        )

    vega_value, vega_coarse = _first_derivative(
        volatility_price, initial_volatility, volatility_bump
    )
    maturity_bump = min(max(maturity * settings.maturity_relative_bump, 1e-5), 0.2 * maturity)
    maturity_derivative, maturity_coarse = _first_derivative(
        lambda bumped: reprice(bumped_maturity=bumped), maturity, maturity_bump
    )
    rho_value, rho_coarse = _first_derivative(
        lambda bumped: reprice(bumped_rate=bumped), rate, settings.rate_bump
    )
    method = f"five-point {engine.upper()} bump"
    return HestonGreekReport(
        engine=f"heston-{engine}",
        price=model_price,
        delta=_estimate(delta_value, delta_coarse, spot_bump, method),
        gamma=_estimate(gamma_value, gamma_coarse, spot_bump, method),
        vega=_estimate(vega_value, vega_coarse, volatility_bump, method),
        theta=_estimate(-maturity_derivative, -maturity_coarse, maturity_bump, method),
        rho=_estimate(rho_value, rho_coarse, settings.rate_bump, method),
        price_evaluations=21,
        vega_variable="parallel shift to sqrt(v0) and sqrt(theta)",
        theta_convention="calendar decay -dV/dT per year",
    )
