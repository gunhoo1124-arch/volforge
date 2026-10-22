"""Cox--Ross--Rubinstein benchmark for American vanilla options."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from volforge.models import black_scholes
from volforge.models.common import OptionType, validate_inputs


@dataclass(frozen=True, slots=True)
class BinomialSettings:
    """CRR exercise-grid resolution."""

    steps: int = 1_000

    def __post_init__(self) -> None:
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("binomial steps must be a positive integer")


@dataclass(frozen=True, slots=True)
class ExerciseBoundaryPoint:
    """Estimated boundary at one exercise time."""

    exercise_time: float
    underlying: float


@dataclass(frozen=True, slots=True)
class BinomialResult:
    """American CRR value and benchmark diagnostics."""

    price: float
    european_price: float
    early_exercise_premium: float
    steps: int
    exercise_boundary: tuple[ExerciseBoundaryPoint, ...]


_DEFAULT_SETTINGS = BinomialSettings()


def american_binomial_price(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    settings: BinomialSettings = _DEFAULT_SETTINGS,
) -> BinomialResult:
    """Price an American vanilla option on a recombining CRR tree."""
    kind = OptionType.parse(option_type)
    validate_inputs(spot, strike, maturity, volatility, rate, dividend_yield)
    european = black_scholes.price(kind, spot, strike, maturity, volatility, rate, dividend_yield)
    intrinsic_now = max(spot - strike, 0.0) if kind is OptionType.CALL else max(strike - spot, 0.0)
    if maturity == 0.0:
        return BinomialResult(intrinsic_now, intrinsic_now, 0.0, settings.steps, ())

    dt = maturity / settings.steps
    if volatility <= 1e-12:
        times = np.linspace(0.0, maturity, settings.steps + 1)
        future_spots = spot * np.exp((rate - dividend_yield) * times)
        payoffs = (
            np.maximum(future_spots - strike, 0.0)
            if kind is OptionType.CALL
            else np.maximum(strike - future_spots, 0.0)
        )
        values = np.exp(-rate * times) * payoffs
        model_price = float(np.max(values))
        return BinomialResult(model_price, european, model_price - european, settings.steps, ())

    up = math.exp(volatility * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp((rate - dividend_yield) * dt)
    probability = (growth - down) / (up - down)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("CRR probability lies outside [0, 1]; increase binomial steps")
    discount = math.exp(-rate * dt)
    nodes = np.arange(settings.steps + 1)
    terminal_spots = spot * np.power(up, 2.0 * nodes - settings.steps)
    values = (
        np.maximum(terminal_spots - strike, 0.0)
        if kind is OptionType.CALL
        else np.maximum(strike - terminal_spots, 0.0)
    )
    boundary: list[ExerciseBoundaryPoint] = []
    tolerance = 1e-12 * max(spot, strike, 1.0)
    for step in range(settings.steps - 1, -1, -1):
        values = discount * (probability * values[1:] + (1.0 - probability) * values[:-1])
        node_numbers = np.arange(step + 1)
        node_spots = spot * np.power(up, 2.0 * node_numbers - step)
        exercise = (
            np.maximum(node_spots - strike, 0.0)
            if kind is OptionType.CALL
            else np.maximum(strike - node_spots, 0.0)
        )
        exercised = (exercise > tolerance) & (exercise >= values - tolerance)
        if np.any(exercised):
            boundary_spot = (
                float(np.min(node_spots[exercised]))
                if kind is OptionType.CALL
                else float(np.max(node_spots[exercised]))
            )
            boundary.append(ExerciseBoundaryPoint(step * dt, boundary_spot))
        values = np.maximum(values, exercise)
    boundary.reverse()
    model_price = float(values[0])
    return BinomialResult(
        model_price,
        european,
        model_price - european,
        settings.steps,
        tuple(boundary),
    )
