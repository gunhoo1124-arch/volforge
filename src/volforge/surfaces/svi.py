"""Constrained raw-SVI smile and surface calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult, minimize

from volforge.surfaces.arbitrage import (
    ButterflyReport,
    CalendarReport,
    butterfly_report,
    calendar_report,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SVIParameters:
    """Raw-SVI parameters for total variance.

    ``w(k) = a + b * (rho * (k-m) + sqrt((k-m)^2 + sigma^2))``.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    @classmethod
    def from_array(cls, values: ArrayLike) -> SVIParameters:
        array = np.asarray(values, dtype=float)
        if array.shape != (5,):
            raise ValueError("SVI requires exactly five parameters")
        return cls(*(float(value) for value in array))

    def as_array(self) -> FloatArray:
        """Return parameters in ``a, b, rho, m, sigma`` order."""
        return np.asarray([self.a, self.b, self.rho, self.m, self.sigma], dtype=float)

    @property
    def minimum_variance(self) -> float:
        """Return the analytic minimum total variance of the raw slice."""
        return self.a + self.b * self.sigma * math.sqrt(1.0 - self.rho**2)

    @property
    def left_wing_slope(self) -> float:
        """Return the asymptotic left total-variance slope magnitude."""
        return self.b * (1.0 - self.rho)

    @property
    def right_wing_slope(self) -> float:
        """Return the asymptotic right total-variance slope."""
        return self.b * (1.0 + self.rho)


@dataclass(frozen=True, slots=True)
class SVISliceCalibration:
    """Structured constrained-calibration result for one expiration."""

    expiration: str
    maturity: float
    parameters: SVIParameters
    optimizer_converged: bool
    iterations: int
    objective: float
    rmse_total_variance: float
    rmse_implied_volatility: float
    quote_count: int
    message: str
    butterfly: ButterflyReport

    @property
    def downstream_eligible(self) -> bool:
        """Whether this slice passed optimization and butterfly checks."""
        return self.optimizer_converged and self.butterfly.is_arbitrage_free


@dataclass(frozen=True, slots=True)
class SVICalibration:
    """A collection of constrained SVI slices plus cross-expiry checks."""

    slices: tuple[SVISliceCalibration, ...]
    calendar: CalendarReport
    rmse_total_variance: float
    rmse_implied_volatility: float
    downstream_eligible: bool
    failure_reasons: tuple[str, ...]

    @property
    def maturities(self) -> tuple[float, ...]:
        """Return calibrated maturities in ascending order."""
        return tuple(slice_result.maturity for slice_result in self.slices)

    def total_variance(self, maturity: float, log_moneyness: ArrayLike) -> FloatArray:
        """Evaluate by linear interpolation in total variance between slices."""
        if not math.isfinite(maturity):
            raise ValueError("maturity must be finite")
        k = np.asarray(log_moneyness, dtype=float)
        maturities = np.asarray(self.maturities, dtype=float)
        if maturity < maturities[0] - 1e-12 or maturity > maturities[-1] + 1e-12:
            raise ValueError("SVI interpolation is limited to calibrated maturities")
        exact = np.flatnonzero(np.isclose(maturities, maturity, rtol=0.0, atol=1e-12))
        if exact.size:
            return svi_total_variance(k, self.slices[int(exact[0])].parameters)
        upper = int(np.searchsorted(maturities, maturity, side="right"))
        lower = upper - 1
        weight = (maturity - maturities[lower]) / (maturities[upper] - maturities[lower])
        near = svi_total_variance(k, self.slices[lower].parameters)
        far = svi_total_variance(k, self.slices[upper].parameters)
        return (1.0 - weight) * near + weight * far

    def evaluated_frame(self, log_moneyness: ArrayLike) -> pd.DataFrame:
        """Return calibrated-node total variance and implied volatility."""
        k = np.asarray(log_moneyness, dtype=float)
        frames: list[pd.DataFrame] = []
        for slice_result in self.slices:
            w = svi_total_variance(k, slice_result.parameters)
            frames.append(
                pd.DataFrame(
                    {
                        "model": "svi",
                        "expiration": slice_result.expiration,
                        "maturity": slice_result.maturity,
                        "log_moneyness": k,
                        "total_variance": w,
                        "implied_volatility": np.sqrt(np.maximum(w, 0.0) / slice_result.maturity),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)


def _validate_parameters(parameters: SVIParameters) -> None:
    values = parameters.as_array()
    if np.any(~np.isfinite(values)):
        raise ValueError("SVI parameters must be finite")
    if parameters.b < 0:
        raise ValueError("SVI b must be non-negative")
    if not -1.0 < parameters.rho < 1.0:
        raise ValueError("SVI rho must be strictly between -1 and 1")
    if parameters.sigma <= 0:
        raise ValueError("SVI sigma must be positive")


def svi_total_variance(log_moneyness: ArrayLike, parameters: SVIParameters) -> FloatArray:
    """Evaluate raw-SVI total implied variance."""
    _validate_parameters(parameters)
    k = np.asarray(log_moneyness, dtype=float)
    if np.any(~np.isfinite(k)):
        raise ValueError("log_moneyness must be finite")
    centered = k - parameters.m
    return parameters.a + parameters.b * (
        parameters.rho * centered + np.sqrt(centered**2 + parameters.sigma**2)
    )


def svi_derivatives(
    log_moneyness: ArrayLike, parameters: SVIParameters
) -> tuple[FloatArray, FloatArray]:
    """Return analytic first and second derivatives of raw-SVI total variance."""
    _validate_parameters(parameters)
    k = np.asarray(log_moneyness, dtype=float)
    if np.any(~np.isfinite(k)):
        raise ValueError("log_moneyness must be finite")
    centered = k - parameters.m
    root = np.sqrt(centered**2 + parameters.sigma**2)
    first = parameters.b * (parameters.rho + centered / root)
    second = parameters.b * parameters.sigma**2 / root**3
    return first, second


def _safe_density(
    k: FloatArray, w: FloatArray, first: FloatArray, second: FloatArray
) -> FloatArray:
    safe_w = np.maximum(w, 1e-10)
    return (
        (1.0 - k * first / (2.0 * safe_w)) ** 2
        - 0.25 * first**2 * (1.0 / safe_w + 0.25)
        + 0.5 * second
    )


def calibrate_svi_slice(
    log_moneyness: ArrayLike,
    total_variance: ArrayLike,
    maturity: float,
    *,
    expiration: str = "",
    weights: ArrayLike | None = None,
    diagnostic_grid: ArrayLike | None = None,
    max_iterations: int = 1500,
) -> SVISliceCalibration:
    """Fit one raw-SVI slice subject to positivity, density, and wing constraints."""
    k = np.asarray(log_moneyness, dtype=float)
    observed = np.asarray(total_variance, dtype=float)
    if k.ndim != 1 or observed.shape != k.shape or k.size < 5:
        raise ValueError("SVI calibration requires at least five paired observations")
    if np.any(~np.isfinite(k)) or np.any(~np.isfinite(observed)) or np.any(observed <= 0):
        raise ValueError("SVI observations must be finite with positive total variance")
    if not math.isfinite(maturity) or maturity <= 0:
        raise ValueError("maturity must be positive and finite")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    order = np.argsort(k)
    k = k[order]
    observed = observed[order]
    if weights is None:
        fit_weights = np.ones_like(observed)
    else:
        fit_weights = np.asarray(weights, dtype=float)[order]
        if (
            fit_weights.shape != observed.shape
            or np.any(~np.isfinite(fit_weights))
            or np.any(fit_weights <= 0)
        ):
            raise ValueError("weights must be finite and positive")
    fit_weights = fit_weights / np.mean(fit_weights)

    if diagnostic_grid is None:
        bound = max(2.0, float(np.max(np.abs(k))) + 0.5)
        grid = np.linspace(-bound, bound, 161)
    else:
        grid = np.asarray(diagnostic_grid, dtype=float)
        if grid.ndim != 1 or grid.size < 21 or np.any(~np.isfinite(grid)):
            raise ValueError("diagnostic_grid must contain at least 21 finite points")

    scale = max(float(np.median(observed)), 1e-6)

    def objective(values: FloatArray) -> float:
        params = SVIParameters.from_array(values)
        fitted = svi_total_variance(k, params)
        residual = (fitted - observed) / scale
        return float(np.mean(fit_weights * residual**2))

    variance_margin = 1e-9
    density_margin = 1e-7

    def constraints(values: FloatArray) -> FloatArray:
        params = SVIParameters.from_array(values)
        w = svi_total_variance(grid, params)
        first, second = svi_derivatives(grid, params)
        g = _safe_density(grid, w, first, second)
        return np.concatenate(
            [
                np.asarray(
                    [
                        params.minimum_variance - variance_margin,
                        2.0 - params.left_wing_slope,
                        2.0 - params.right_wing_slope,
                    ]
                ),
                w - variance_margin,
                g - density_margin,
            ]
        )

    spread = max(float(np.ptp(k)), 0.1)
    variance_range = max(float(np.ptp(observed)), scale * 0.05)
    base_b = min(max(variance_range / spread, 1e-4), 0.5)
    minimum_index = int(np.argmin(observed))
    base_m = float(k[minimum_index])
    base_sigma = max(min(spread / 2.0, 0.6), 0.08)
    starts: list[FloatArray] = []
    for rho in (-0.6, -0.3, 0.0, 0.3):
        for sigma_multiplier in (0.7, 1.5):
            sigma = min(max(base_sigma * sigma_multiplier, 0.03), 1.5)
            a = max(
                float(np.min(observed)) - base_b * sigma * math.sqrt(1.0 - rho**2),
                1e-6,
            )
            starts.append(np.asarray([a, base_b, rho, base_m, sigma], dtype=float))

    max_variance = float(np.max(observed))
    bounds = [
        (-max(0.25, 4.0 * max_variance), max(0.5, 6.0 * max_variance)),
        (1e-8, 1.999),
        (-0.999, 0.999),
        (float(np.min(k)) - 2.0, float(np.max(k)) + 2.0),
        (1e-4, 3.0),
    ]
    best: OptimizeResult | None = None
    best_rank = (2, math.inf)
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": max_iterations, "ftol": 1e-12, "disp": False},
        )
        minimum_constraint = float(np.min(constraints(np.asarray(result.x, dtype=float))))
        feasible = minimum_constraint >= -2e-6
        rank = (0 if result.success and feasible else 1 if feasible else 2, float(result.fun))
        if rank < best_rank:
            best = result
            best_rank = rank

    if best is None:
        raise RuntimeError("SVI optimizer produced no result")
    parameters = SVIParameters.from_array(best.x)
    fitted = svi_total_variance(k, parameters)
    first, second = svi_derivatives(grid, parameters)
    grid_variance = svi_total_variance(grid, parameters)
    report = butterfly_report(
        grid,
        grid_variance,
        first,
        second,
        left_wing_slope=parameters.left_wing_slope,
        right_wing_slope=parameters.right_wing_slope,
        tolerance=2e-6,
    )
    fitted_iv = np.sqrt(np.maximum(fitted, 0.0) / maturity)
    observed_iv = np.sqrt(observed / maturity)
    feasible = float(np.min(constraints(np.asarray(best.x, dtype=float)))) >= -2e-6
    return SVISliceCalibration(
        expiration=expiration,
        maturity=float(maturity),
        parameters=parameters,
        optimizer_converged=bool(best.success and feasible),
        iterations=int(best.nit),
        objective=float(best.fun),
        rmse_total_variance=float(np.sqrt(np.mean((fitted - observed) ** 2))),
        rmse_implied_volatility=float(np.sqrt(np.mean((fitted_iv - observed_iv) ** 2))),
        quote_count=len(k),
        message=str(best.message),
        butterfly=report,
    )


def calibrate_svi_surface(
    quotes: pd.DataFrame,
    *,
    diagnostic_grid: ArrayLike | None = None,
    max_iterations: int = 1500,
) -> SVICalibration:
    """Fit constrained SVI slices and diagnose calendar spread arbitrage."""
    required = {"expiration", "maturity", "log_moneyness", "total_variance", "quote_count"}
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"surface quotes are missing: {', '.join(sorted(missing))}")
    if quotes["expiration"].nunique() < 2:
        raise ValueError("SVI surface calibration requires at least two expirations")
    if diagnostic_grid is None:
        largest = max(2.0, float(quotes["log_moneyness"].abs().max()) + 0.5)
        grid = np.linspace(-largest, largest, 161)
    else:
        grid = np.asarray(diagnostic_grid, dtype=float)

    slice_results: list[SVISliceCalibration] = []
    ordered = quotes.sort_values(["maturity", "log_moneyness"])
    for expiration, group in ordered.groupby("expiration", sort=False):
        maturity = float(group["maturity"].median())
        slice_results.append(
            calibrate_svi_slice(
                group["log_moneyness"].to_numpy(),
                group["total_variance"].to_numpy(),
                maturity,
                expiration=str(expiration),
                weights=group["quote_count"].to_numpy(),
                diagnostic_grid=grid,
                max_iterations=max_iterations,
            )
        )
    slice_results.sort(key=lambda result: result.maturity)

    def evaluator(maturity: float, k: FloatArray) -> FloatArray:
        index = next(
            i
            for i, result in enumerate(slice_results)
            if math.isclose(result.maturity, maturity, rel_tol=0.0, abs_tol=1e-12)
        )
        return svi_total_variance(k, slice_results[index].parameters)

    calendar = calendar_report(
        [result.maturity for result in slice_results], evaluator, grid, tolerance=2e-6
    )
    quote_total = sum(result.quote_count for result in slice_results)
    rmse_variance = math.sqrt(
        sum(result.quote_count * result.rmse_total_variance**2 for result in slice_results)
        / quote_total
    )
    rmse_iv = math.sqrt(
        sum(result.quote_count * result.rmse_implied_volatility**2 for result in slice_results)
        / quote_total
    )
    reasons: list[str] = []
    for result in slice_results:
        if not result.optimizer_converged:
            reasons.append(f"SVI optimizer failed for {result.expiration}: {result.message}")
        if not result.butterfly.is_arbitrage_free:
            reasons.append(f"butterfly diagnostic failed for {result.expiration}")
    if not calendar.is_arbitrage_free:
        reasons.append("calendar-spread diagnostic failed between SVI slices")
    return SVICalibration(
        slices=tuple(slice_results),
        calendar=calendar,
        rmse_total_variance=rmse_variance,
        rmse_implied_volatility=rmse_iv,
        downstream_eligible=not reasons,
        failure_reasons=tuple(reasons),
    )
