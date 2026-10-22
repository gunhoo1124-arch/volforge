"""Heston American-option obstacle solver with Ikonen--Toivanen splitting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from volforge.american.binomial import ExerciseBoundaryPoint
from volforge.models.common import OptionType
from volforge.models.heston import HestonParameters
from volforge.numerics.pde import (
    PDEGridSettings,
    _bilinear_value,
    _operator,
    _set_constraints,
    _validate_contract,
    heston_pde_price,
)


@dataclass(frozen=True, slots=True)
class AmericanPDEResult:
    """American Heston PDE value and free-boundary diagnostics."""

    price: float
    european_price: float
    early_exercise_premium: float
    exercise_fraction: float
    log_price_points: int
    variance_points: int
    time_steps: int
    log_price_min: float
    log_price_max: float
    variance_max: float
    rannacher_steps: int
    method: str
    exercise_boundary: tuple[ExerciseBoundaryPoint, ...]


_DEFAULT_SETTINGS = PDEGridSettings()


def _payoff(kind: OptionType, underlying: np.ndarray, strike: float) -> np.ndarray:
    if kind is OptionType.CALL:
        return np.maximum(underlying - strike, 0.0)
    return np.maximum(strike - underlying, 0.0)


def _dirichlet_values(
    kind: OptionType,
    tau: float,
    underlying: np.ndarray,
    strike: float,
    rate: float,
    dividend_yield: float,
) -> tuple[float, float]:
    if kind is OptionType.PUT:
        left = max(
            strike - underlying[0],
            strike * math.exp(-rate * tau) - underlying[0] * math.exp(-dividend_yield * tau),
            0.0,
        )
        return left, 0.0
    right = max(
        underlying[-1] - strike,
        underlying[-1] * math.exp(-dividend_yield * tau) - strike * math.exp(-rate * tau),
        0.0,
    )
    return 0.0, right


def _set_rhs_boundaries(
    values: np.ndarray,
    kind: OptionType,
    tau: float,
    underlying: np.ndarray,
    strike: float,
    rate: float,
    dividend_yield: float,
    nx: int,
    nv: int,
) -> None:
    left, right = _dirichlet_values(kind, tau, underlying, strike, rate, dividend_yield)
    for j in range(nv):
        values[j * nx] = left
        values[j * nx + nx - 1] = right
    for i in range(1, nx - 1):
        values[(nv - 1) * nx + i] = 0.0


def _set_value_boundaries(
    values: np.ndarray,
    kind: OptionType,
    tau: float,
    underlying: np.ndarray,
    strike: float,
    rate: float,
    dividend_yield: float,
    nx: int,
    nv: int,
) -> None:
    left, right = _dirichlet_values(kind, tau, underlying, strike, rate, dividend_yield)
    grid = values.reshape((nv, nx))
    grid[:, 0] = left
    grid[:, -1] = right
    grid[-1, 1:-1] = grid[-2, 1:-1]


def _it_projection(
    predictor: np.ndarray,
    multiplier: np.ndarray,
    obstacle: np.ndarray,
    step_size: float,
    active_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = predictor.copy()
    values[active_nodes] = np.maximum(
        predictor[active_nodes] - step_size * multiplier[active_nodes],
        obstacle[active_nodes],
    )
    next_multiplier = np.zeros_like(multiplier)
    next_multiplier[active_nodes] = np.maximum(
        0.0,
        multiplier[active_nodes] + (obstacle[active_nodes] - predictor[active_nodes]) / step_size,
    )
    return values, next_multiplier


def heston_american_pde_price(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    settings: PDEGridSettings = _DEFAULT_SETTINGS,
) -> AmericanPDEResult:
    """Price an American option with CN/BE Ikonen--Toivanen splitting."""
    kind = OptionType.parse(option_type)
    _validate_contract(spot, strike, maturity, rate, dividend_yield)
    if parameters.initial_variance >= settings.variance_max:
        raise ValueError("PDE variance_max must exceed the initial variance")
    x_center = math.log(spot / strike)
    x_coordinate = np.linspace(-1.0, 1.0, settings.log_price_points)
    x_points = x_center + settings.log_price_half_width * np.sinh(2.5 * x_coordinate) / math.sinh(
        2.5
    )
    variance_coordinate = np.linspace(0.0, 1.0, settings.variance_points)
    variance_points = settings.variance_max * np.sinh(3.0 * variance_coordinate) / math.sinh(3.0)
    underlying = strike * np.exp(x_points)
    nx = settings.log_price_points
    nv = settings.variance_points
    obstacle = np.tile(_payoff(kind, underlying, strike), nv)
    if maturity == 0.0:
        intrinsic = float(_payoff(kind, np.asarray([spot]), strike)[0])
        return AmericanPDEResult(
            intrinsic,
            intrinsic,
            0.0,
            0.0,
            nx,
            nv,
            settings.time_steps,
            float(x_points[0]),
            float(x_points[-1]),
            settings.variance_max,
            settings.rannacher_steps,
            "Ikonen-Toivanen CN",
            (),
        )

    operator = _operator(x_points, variance_points, parameters, rate, dividend_yield)
    identity = sparse.identity(nx * nv, format="csr")
    time_step = maturity / settings.time_steps
    crank_left = (identity - 0.5 * time_step * operator).tolil()
    crank_right = (identity + 0.5 * time_step * operator).tolil()
    _set_constraints(crank_left, crank_right, nx, nv)
    crank_solve = factorized(crank_left.tocsc())
    crank_right = crank_right.tocsr()
    half_step = 0.5 * time_step
    implicit_left = (identity - half_step * operator).tolil()
    _set_constraints(implicit_left, None, nx, nv)
    implicit_solve = factorized(implicit_left.tocsc())

    active_nodes = np.zeros(nx * nv, dtype=bool)
    for j in range(nv - 1):
        active_nodes[j * nx + 1 : j * nx + nx - 1] = True
    values = obstacle.copy()
    multiplier = np.zeros_like(values)
    _set_value_boundaries(values, kind, 0.0, underlying, strike, rate, dividend_yield, nx, nv)
    tau = 0.0
    damping_intervals = min(settings.rannacher_steps, settings.time_steps)
    boundary: list[ExerciseBoundaryPoint] = []
    variance_index = int(np.argmin(np.abs(variance_points - parameters.initial_variance)))

    for step in range(settings.time_steps):
        if step < damping_intervals:
            for _ in range(2):
                tau += half_step
                rhs = values + half_step * multiplier
                _set_rhs_boundaries(
                    rhs, kind, tau, underlying, strike, rate, dividend_yield, nx, nv
                )
                predictor = np.asarray(implicit_solve(rhs), dtype=float)
                values, multiplier = _it_projection(
                    predictor, multiplier, obstacle, half_step, active_nodes
                )
                _set_value_boundaries(
                    values, kind, tau, underlying, strike, rate, dividend_yield, nx, nv
                )
        else:
            tau += time_step
            rhs = np.asarray(crank_right @ values, dtype=float) + time_step * multiplier
            _set_rhs_boundaries(rhs, kind, tau, underlying, strike, rate, dividend_yield, nx, nv)
            predictor = np.asarray(crank_solve(rhs), dtype=float)
            values, multiplier = _it_projection(
                predictor, multiplier, obstacle, time_step, active_nodes
            )
            _set_value_boundaries(
                values, kind, tau, underlying, strike, rate, dividend_yield, nx, nv
            )

        row = values.reshape((nv, nx))[variance_index]
        intrinsic = _payoff(kind, underlying, strike)
        exercised = (intrinsic > 0.0) & (row <= intrinsic + 1e-7 * max(strike, 1.0))
        if np.any(exercised):
            boundary_spot = (
                float(np.min(underlying[exercised]))
                if kind is OptionType.CALL
                else float(np.max(underlying[exercised]))
            )
            boundary.append(ExerciseBoundaryPoint(max(maturity - tau, 0.0), boundary_spot))

    model_price = _bilinear_value(
        values, x_points, variance_points, x_center, parameters.initial_variance
    )
    european_price = heston_pde_price(
        kind,
        spot,
        strike,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=settings,
    ).price
    if kind is OptionType.CALL and dividend_yield <= 0.0 and rate >= 0.0:
        model_price = european_price
        multiplier.fill(0.0)
        boundary.clear()
    exercise_fraction = float(np.mean((multiplier > 0.0)[active_nodes]))
    if not math.isfinite(model_price) or model_price < -1e-8:
        raise ArithmeticError("American PDE produced an invalid value")
    reported_price = max(model_price, float(_payoff(kind, np.asarray([spot]), strike)[0]))
    return AmericanPDEResult(
        reported_price,
        european_price,
        reported_price - european_price,
        exercise_fraction,
        nx,
        nv,
        settings.time_steps,
        float(x_points[0]),
        float(x_points[-1]),
        settings.variance_max,
        settings.rannacher_steps,
        "Ikonen-Toivanen CN",
        tuple(boundary),
    )
