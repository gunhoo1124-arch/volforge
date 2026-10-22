"""Finite-difference valuation for the two-dimensional Heston PDE."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import factorized

from volforge.models.common import OptionType, discounted_price_bounds
from volforge.models.heston import HestonParameters


@dataclass(frozen=True, slots=True)
class PDEGridSettings:
    """Non-uniform log-price/variance grid and time-stepping controls."""

    log_price_points: int = 121
    variance_points: int = 61
    time_steps: int = 180
    log_price_half_width: float = 4.0
    variance_max: float = 1.0
    rannacher_steps: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.log_price_points, int) or self.log_price_points < 21:
            raise ValueError("PDE log_price_points must be an integer of at least 21")
        if not isinstance(self.variance_points, int) or self.variance_points < 11:
            raise ValueError("PDE variance_points must be an integer of at least 11")
        if not isinstance(self.time_steps, int) or self.time_steps < 1:
            raise ValueError("PDE time_steps must be a positive integer")
        if not math.isfinite(self.log_price_half_width) or self.log_price_half_width <= 0.5:
            raise ValueError("PDE log_price_half_width must exceed 0.5")
        if not math.isfinite(self.variance_max) or self.variance_max <= 0.0:
            raise ValueError("PDE variance_max must be positive")
        if not isinstance(self.rannacher_steps, int) or not 0 <= self.rannacher_steps <= 4:
            raise ValueError("PDE rannacher_steps must be an integer from zero to four")


@dataclass(frozen=True, slots=True)
class PDEPriceResult:
    """A finite-difference price and the domain used to obtain it."""

    price: float
    lower_bound: float
    upper_bound: float
    log_price_points: int
    variance_points: int
    time_steps: int
    log_price_min: float
    log_price_max: float
    variance_max: float
    log_price_spacing: float
    variance_spacing: float
    rannacher_steps: int


_DEFAULT_PDE_SETTINGS = PDEGridSettings()


def _validate_contract(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
) -> None:
    values = (spot, strike, maturity, rate, dividend_yield)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("contract inputs must be finite numbers")
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")


def _operator(
    x_points: np.ndarray,
    variance_points: np.ndarray,
    parameters: HestonParameters,
    rate: float,
    dividend_yield: float,
) -> sparse.csr_matrix:
    nx = len(x_points)
    nv = len(variance_points)
    matrix = sparse.lil_matrix((nx * nv, nx * nv), dtype=float)

    def index(variance_index: int, price_index: int) -> int:
        return variance_index * nx + price_index

    def weights(points: np.ndarray, point_index: int) -> tuple[np.ndarray, np.ndarray]:
        lower = float(points[point_index] - points[point_index - 1])
        upper = float(points[point_index + 1] - points[point_index])
        first = np.asarray(
            [
                -upper / (lower * (lower + upper)),
                (upper - lower) / (lower * upper),
                lower / (upper * (lower + upper)),
            ]
        )
        second = np.asarray(
            [
                2.0 / (lower * (lower + upper)),
                -2.0 / (lower * upper),
                2.0 / (upper * (lower + upper)),
            ]
        )
        return first, second

    for j, variance in enumerate(variance_points[:-1]):
        for i in range(1, nx - 1):
            row = index(j, i)
            x_drift = rate - dividend_yield - 0.5 * variance
            x_first, x_second = weights(x_points, i)
            for x_offset, (first_weight, second_weight) in enumerate(
                zip(x_first, x_second, strict=True), start=-1
            ):
                matrix[row, index(j, i + x_offset)] += (
                    x_drift * first_weight + 0.5 * variance * second_weight
                )
            matrix[row, row] -= rate

            variance_drift = parameters.kappa * (parameters.theta - variance)
            if j == 0:
                first_spacing = float(variance_points[1] - variance_points[0])
                matrix[row, index(j, i)] += -variance_drift / first_spacing
                matrix[row, index(j + 1, i)] += variance_drift / first_spacing
                continue

            variance_diffusion = 0.5 * parameters.vol_of_vol**2 * variance
            variance_first, variance_second = weights(variance_points, j)
            for variance_offset, (first_weight, second_weight) in enumerate(
                zip(variance_first, variance_second, strict=True), start=-1
            ):
                matrix[row, index(j + variance_offset, i)] += (
                    variance_drift * first_weight + variance_diffusion * second_weight
                )

            cross = parameters.rho * parameters.vol_of_vol * variance
            for variance_offset, variance_weight in enumerate(variance_first, start=-1):
                for x_offset, x_weight in enumerate(x_first, start=-1):
                    matrix[row, index(j + variance_offset, i + x_offset)] += (
                        cross * variance_weight * x_weight
                    )
    return matrix.tocsr()


def _set_constraints(
    left: sparse.lil_matrix,
    right: sparse.lil_matrix | None,
    nx: int,
    nv: int,
) -> None:
    for j in range(nv):
        for i in (0, nx - 1):
            row = j * nx + i
            left.rows[row] = [row]
            left.data[row] = [1.0]
            if right is not None:
                right.rows[row] = [row]
                right.data[row] = [0.0]
    for i in range(1, nx - 1):
        row = (nv - 1) * nx + i
        left.rows[row] = [row - nx, row]
        left.data[row] = [-1.0, 1.0]
        if right is not None:
            right.rows[row] = []
            right.data[row] = []


def _boundary_values(
    values: np.ndarray,
    tau: float,
    x_points: np.ndarray,
    strike: float,
    rate: float,
    dividend_yield: float,
) -> None:
    nx = len(x_points)
    nv = values.size // nx
    left_value = max(
        strike * math.exp(-rate * tau)
        - strike * math.exp(x_points[0]) * math.exp(-dividend_yield * tau),
        0.0,
    )
    for j in range(nv):
        values[j * nx] = left_value
        values[j * nx + nx - 1] = 0.0
    for i in range(1, nx - 1):
        values[(nv - 1) * nx + i] = 0.0


def _bilinear_value(
    values: np.ndarray,
    x_points: np.ndarray,
    variance_points: np.ndarray,
    x: float,
    variance: float,
) -> float:
    nx = len(x_points)
    grid = values.reshape((len(variance_points), nx))
    x_index = min(max(int(np.searchsorted(x_points, x)) - 1, 0), nx - 2)
    v_index = min(
        max(int(np.searchsorted(variance_points, variance)) - 1, 0),
        len(variance_points) - 2,
    )
    x_weight = (x - x_points[x_index]) / (x_points[x_index + 1] - x_points[x_index])
    v_weight = (variance - variance_points[v_index]) / (
        variance_points[v_index + 1] - variance_points[v_index]
    )
    lower = (1.0 - x_weight) * grid[v_index, x_index] + x_weight * grid[v_index, x_index + 1]
    upper = (1.0 - x_weight) * grid[v_index + 1, x_index] + x_weight * grid[
        v_index + 1, x_index + 1
    ]
    return float((1.0 - v_weight) * lower + v_weight * upper)


def _cubic_linear_value(
    values: np.ndarray,
    x_points: np.ndarray,
    variance_points: np.ndarray,
    x: float,
    variance: float,
) -> float:
    """Interpolate cubically in log-price and linearly in variance."""
    nx = len(x_points)
    grid = values.reshape((len(variance_points), nx))
    v_index = min(
        max(int(np.searchsorted(variance_points, variance)) - 1, 0),
        len(variance_points) - 2,
    )
    v_weight = (variance - variance_points[v_index]) / (
        variance_points[v_index + 1] - variance_points[v_index]
    )
    lower = float(CubicSpline(x_points, grid[v_index], bc_type="natural")(x))
    upper = float(CubicSpline(x_points, grid[v_index + 1], bc_type="natural")(x))
    return (1.0 - v_weight) * lower + v_weight * upper


def heston_pde_price(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    settings: PDEGridSettings = _DEFAULT_PDE_SETTINGS,
    grid_center_spot: float | None = None,
    interpolation: str = "linear",
) -> PDEPriceResult:
    """Price a European option on a bounded grid with Crank--Nicolson.

    The solver evolves a put value in time-to-maturity and obtains calls by
    put--call parity. Two half-sized implicit steps damp the payoff kink during
    each configured Rannacher interval.
    """
    kind = OptionType.parse(option_type)
    _validate_contract(spot, strike, maturity, rate, dividend_yield)
    lower_bound, upper_bound = discounted_price_bounds(
        kind, spot, strike, maturity, rate, dividend_yield
    )
    if grid_center_spot is not None and (
        not math.isfinite(grid_center_spot) or grid_center_spot <= 0.0
    ):
        raise ValueError("PDE grid_center_spot must be positive and finite")
    if interpolation not in {"linear", "cubic"}:
        raise ValueError("PDE interpolation must be 'linear' or 'cubic'")
    evaluation_x = math.log(spot / strike)
    grid_center_x = math.log((grid_center_spot or spot) / strike)
    x_min = grid_center_x - settings.log_price_half_width
    x_max = grid_center_x + settings.log_price_half_width
    if parameters.initial_variance >= settings.variance_max:
        raise ValueError("PDE variance_max must exceed the initial variance")
    if maturity == 0.0:
        intrinsic = max(spot - strike, 0.0) if kind is OptionType.CALL else max(strike - spot, 0.0)
        return PDEPriceResult(
            intrinsic,
            lower_bound,
            upper_bound,
            settings.log_price_points,
            settings.variance_points,
            settings.time_steps,
            x_min,
            x_max,
            settings.variance_max,
            2.0 * settings.log_price_half_width / (settings.log_price_points - 1),
            settings.variance_max / (settings.variance_points - 1),
            settings.rannacher_steps,
        )

    x_coordinate = np.linspace(-1.0, 1.0, settings.log_price_points)
    x_concentration = 2.5
    x_points = grid_center_x + settings.log_price_half_width * np.sinh(
        x_concentration * x_coordinate
    ) / math.sinh(x_concentration)
    variance_coordinate = np.linspace(0.0, 1.0, settings.variance_points)
    variance_concentration = 3.0
    variance_points = (
        settings.variance_max
        * np.sinh(variance_concentration * variance_coordinate)
        / math.sinh(variance_concentration)
    )
    nx = settings.log_price_points
    nv = settings.variance_points
    operator = _operator(x_points, variance_points, parameters, rate, dividend_yield)
    identity = sparse.identity(nx * nv, format="csr")
    time_step = maturity / settings.time_steps

    crank_left = (identity - 0.5 * time_step * operator).tolil()
    crank_right = (identity + 0.5 * time_step * operator).tolil()
    _set_constraints(crank_left, crank_right, nx, nv)
    crank_solve = factorized(crank_left.tocsc())
    crank_right_csr = crank_right.tocsr()

    half_step = 0.5 * time_step
    implicit_left = (identity - half_step * operator).tolil()
    _set_constraints(implicit_left, None, nx, nv)
    implicit_solve = factorized(implicit_left.tocsc())

    payoff = strike * np.maximum(1.0 - np.exp(x_points), 0.0)
    values = np.tile(payoff, nv)
    _boundary_values(values, 0.0, x_points, strike, rate, dividend_yield)
    tau = 0.0
    damping_intervals = min(settings.rannacher_steps, settings.time_steps)
    for step in range(settings.time_steps):
        if step < damping_intervals:
            for _ in range(2):
                tau += half_step
                right_hand_side = values.copy()
                _boundary_values(right_hand_side, tau, x_points, strike, rate, dividend_yield)
                values = np.asarray(implicit_solve(right_hand_side), dtype=float)
        else:
            tau += time_step
            right_hand_side = np.asarray(crank_right_csr @ values, dtype=float)
            _boundary_values(right_hand_side, tau, x_points, strike, rate, dividend_yield)
            values = np.asarray(crank_solve(right_hand_side), dtype=float)

    interpolate = _bilinear_value if interpolation == "linear" else _cubic_linear_value
    put_price = interpolate(
        values, x_points, variance_points, evaluation_x, parameters.initial_variance
    )
    model_price = (
        put_price
        if kind is OptionType.PUT
        else put_price
        + spot * math.exp(-dividend_yield * maturity)
        - strike * math.exp(-rate * maturity)
    )
    tolerance = 5e-3 * max(1.0, upper_bound)
    if (
        not math.isfinite(model_price)
        or model_price < lower_bound - tolerance
        or model_price > upper_bound + tolerance
    ):
        raise ArithmeticError("PDE pricing produced a non-finite or out-of-bounds value")
    model_price = min(max(model_price, lower_bound), upper_bound)
    return PDEPriceResult(
        price=model_price,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        log_price_points=nx,
        variance_points=nv,
        time_steps=settings.time_steps,
        log_price_min=x_min,
        log_price_max=x_max,
        variance_max=settings.variance_max,
        log_price_spacing=float(np.min(np.diff(x_points))),
        variance_spacing=float(np.min(np.diff(variance_points))),
        rannacher_steps=settings.rannacher_steps,
    )
