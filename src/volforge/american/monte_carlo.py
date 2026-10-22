"""Longstaff--Schwartz American-option valuation under Heston dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from volforge.american.binomial import ExerciseBoundaryPoint
from volforge.models.common import OptionType
from volforge.models.heston import HestonParameters
from volforge.numerics.monte_carlo import HestonPathSettings, simulate_heston_paths


@dataclass(frozen=True, slots=True)
class LSMSettings:
    """Simulation and regression controls for least-squares Monte Carlo."""

    paths: int = 20_000
    steps: int = 64
    seed: int = 7
    antithetic: bool = True
    psi_threshold: float = 1.5

    def __post_init__(self) -> None:
        HestonPathSettings(self.paths, self.steps, self.seed, self.antithetic, self.psi_threshold)


@dataclass(frozen=True, slots=True)
class AmericanMonteCarloResult:
    """LSM lower-bound estimate with conditional sampling diagnostics."""

    price: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    european_price: float
    early_exercise_premium: float
    exercise_fraction: float
    average_exercise_time: float | None
    paths: int
    steps: int
    seed: int
    antithetic: bool
    basis: str
    exercise_boundary: tuple[ExerciseBoundaryPoint, ...]


_DEFAULT_SETTINGS = LSMSettings()


def _payoff(kind: OptionType, spots: np.ndarray, strike: float) -> np.ndarray:
    if kind is OptionType.CALL:
        return np.maximum(spots - strike, 0.0)
    return np.maximum(strike - spots, 0.0)


def _basis(spots: np.ndarray, variances: np.ndarray, strike: float, theta: float) -> np.ndarray:
    x = spots / strike
    v = variances / max(theta, 1e-8)
    return np.column_stack((np.ones_like(x), x, x * x, v, v * v, x * v))


def _independent_observations(values: np.ndarray, antithetic: bool) -> np.ndarray:
    if not antithetic:
        return values
    half = len(values) // 2
    return 0.5 * (values[:half] + values[half:])


def heston_lsm_price(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    settings: LSMSettings = _DEFAULT_SETTINGS,
) -> AmericanMonteCarloResult:
    """Estimate an American value using Longstaff--Schwartz regression."""
    kind = OptionType.parse(option_type)
    simulation = simulate_heston_paths(
        spot,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=HestonPathSettings(
            settings.paths,
            settings.steps,
            settings.seed,
            settings.antithetic,
            settings.psi_threshold,
        ),
    )
    intrinsic_now = float(_payoff(kind, np.asarray([spot]), strike)[0])
    if maturity == 0.0:
        return AmericanMonteCarloResult(
            intrinsic_now,
            0.0,
            intrinsic_now,
            intrinsic_now,
            intrinsic_now,
            0.0,
            0.0,
            None,
            settings.paths,
            settings.steps,
            settings.seed,
            settings.antithetic,
            "1,S,S^2,v,v^2,Sv",
            (),
        )

    discount = math.exp(-rate * simulation.time_step)
    terminal_payoff = _payoff(kind, simulation.spots[-1], strike)
    european_observations = math.exp(-rate * maturity) * terminal_payoff
    european_price = float(
        _independent_observations(european_observations, settings.antithetic).mean()
    )
    cashflows = terminal_payoff.copy()
    exercise_steps = np.full(settings.paths, settings.steps, dtype=int)
    boundary: list[ExerciseBoundaryPoint] = []
    allow_early_exercise = not (kind is OptionType.CALL and dividend_yield <= 0.0 and rate >= 0.0)

    for step in range(settings.steps - 1, 0, -1):
        cashflows *= discount
        if not allow_early_exercise:
            continue
        exercise = _payoff(kind, simulation.spots[step], strike)
        in_money = exercise > 0.0
        if int(np.count_nonzero(in_money)) < 6:
            continue
        design = _basis(
            simulation.spots[step, in_money],
            simulation.variances[step, in_money],
            strike,
            parameters.theta,
        )
        coefficients, *_ = np.linalg.lstsq(design, cashflows[in_money], rcond=None)
        continuation = design @ coefficients
        local_exercise = exercise[in_money] > continuation
        selected = np.flatnonzero(in_money)[local_exercise]
        if selected.size:
            cashflows[selected] = exercise[selected]
            exercise_steps[selected] = step
            selected_spots = simulation.spots[step, selected]
            boundary_spot = (
                float(np.min(selected_spots))
                if kind is OptionType.CALL
                else float(np.max(selected_spots))
            )
            boundary.append(ExerciseBoundaryPoint(step * simulation.time_step, boundary_spot))

    discounted_cashflows = discount * cashflows
    observations = _independent_observations(discounted_cashflows, settings.antithetic)
    model_price = float(observations.mean())
    standard_error = float(observations.std(ddof=1) / math.sqrt(len(observations)))
    half_width = 1.959963984540054 * standard_error
    early = exercise_steps < settings.steps
    exercise_fraction = float(np.mean(early))
    average_exercise_time = (
        float(np.mean(exercise_steps[early]) * simulation.time_step) if np.any(early) else None
    )
    boundary.reverse()
    if not math.isfinite(model_price) or not math.isfinite(standard_error):
        raise ArithmeticError("LSM produced a non-finite estimate")
    return AmericanMonteCarloResult(
        model_price,
        standard_error,
        model_price - half_width,
        model_price + half_width,
        european_price,
        model_price - european_price,
        exercise_fraction,
        average_exercise_time,
        settings.paths,
        settings.steps,
        settings.seed,
        settings.antithetic,
        "1,S/K,(S/K)^2,v/theta,(v/theta)^2,(S/K)(v/theta)",
        tuple(boundary),
    )
