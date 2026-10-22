"""Power-law SSVI calibration with analytic static-arbitrage constraints."""

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
class SSVIParameters:
    """Power-law SSVI parameters and calibrated ATM variance nodes."""

    rho: float
    eta: float
    gamma: float
    maturities: tuple[float, ...]
    theta: tuple[float, ...]

    def __post_init__(self) -> None:
        if not -1.0 < self.rho < 1.0:
            raise ValueError("SSVI rho must be strictly between -1 and 1")
        if not math.isfinite(self.eta) or self.eta <= 0:
            raise ValueError("SSVI eta must be positive and finite")
        if not math.isfinite(self.gamma) or not 0.0 < self.gamma < 1.0:
            raise ValueError("SSVI gamma must be strictly between zero and one")
        maturities = np.asarray(self.maturities, dtype=float)
        theta = np.asarray(self.theta, dtype=float)
        if maturities.ndim != 1 or len(maturities) < 2 or theta.shape != maturities.shape:
            raise ValueError("SSVI requires matching theta values for at least two maturities")
        if np.any(~np.isfinite(maturities)) or np.any(np.diff(maturities) <= 0):
            raise ValueError("SSVI maturities must be finite and strictly increasing")
        if np.any(~np.isfinite(theta)) or np.any(theta <= 0) or np.any(np.diff(theta) < 0):
            raise ValueError("SSVI theta nodes must be finite, positive, and nondecreasing")

    def theta_at(self, maturity: float) -> float:
        """Linearly interpolate ATM total variance between calibrated maturities."""
        if not math.isfinite(maturity) or maturity <= 0:
            raise ValueError("maturity must be positive and finite")
        maturities = np.asarray(self.maturities, dtype=float)
        theta = np.asarray(self.theta, dtype=float)
        if maturity < maturities[0] - 1e-12 or maturity > maturities[-1] + 1e-12:
            raise ValueError("SSVI evaluation is limited to calibrated maturities")
        return float(np.interp(maturity, maturities, theta))


@dataclass(frozen=True, slots=True)
class SSVICalibration:
    """Structured joint SSVI calibration and arbitrage diagnostics."""

    parameters: SSVIParameters
    expirations: tuple[str, ...]
    optimizer_converged: bool
    iterations: int
    objective: float
    rmse_total_variance: float
    rmse_implied_volatility: float
    quote_count: int
    message: str
    butterflies: tuple[ButterflyReport, ...]
    calendar: CalendarReport
    theorem_conditions_satisfied: bool
    downstream_eligible: bool
    failure_reasons: tuple[str, ...]

    @property
    def maturities(self) -> tuple[float, ...]:
        """Return calibrated maturity nodes."""
        return self.parameters.maturities

    def total_variance(self, maturity: float, log_moneyness: ArrayLike) -> FloatArray:
        """Evaluate SSVI within the calibrated maturity horizon."""
        theta = self.parameters.theta_at(maturity)
        return ssvi_total_variance(
            log_moneyness,
            theta,
            self.parameters.rho,
            self.parameters.eta,
            self.parameters.gamma,
        )

    def evaluated_frame(self, log_moneyness: ArrayLike) -> pd.DataFrame:
        """Return calibrated-node total variance and implied volatility."""
        k = np.asarray(log_moneyness, dtype=float)
        frames: list[pd.DataFrame] = []
        for expiration, maturity, theta in zip(
            self.expirations,
            self.parameters.maturities,
            self.parameters.theta,
            strict=True,
        ):
            w = ssvi_total_variance(
                k,
                theta,
                self.parameters.rho,
                self.parameters.eta,
                self.parameters.gamma,
            )
            frames.append(
                pd.DataFrame(
                    {
                        "model": "ssvi",
                        "expiration": expiration,
                        "maturity": maturity,
                        "log_moneyness": k,
                        "total_variance": w,
                        "implied_volatility": np.sqrt(np.maximum(w, 0.0) / maturity),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)


def power_law_phi(theta: ArrayLike, eta: float, gamma: float) -> FloatArray:
    """Return ``phi(theta) = eta * theta**(-gamma)``."""
    theta_array = np.asarray(theta, dtype=float)
    if np.any(~np.isfinite(theta_array)) or np.any(theta_array <= 0):
        raise ValueError("theta must be finite and positive")
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be positive and finite")
    if not math.isfinite(gamma) or not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be strictly between zero and one")
    return eta * theta_array ** (-gamma)


def ssvi_total_variance(
    log_moneyness: ArrayLike,
    theta: float,
    rho: float,
    eta: float,
    gamma: float,
) -> FloatArray:
    """Evaluate power-law SSVI total implied variance."""
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be strictly between -1 and 1")
    k = np.asarray(log_moneyness, dtype=float)
    if np.any(~np.isfinite(k)):
        raise ValueError("log_moneyness must be finite")
    phi = float(power_law_phi(np.asarray([theta]), eta, gamma)[0])
    shifted = phi * k + rho
    root = np.sqrt(shifted**2 + 1.0 - rho**2)
    return 0.5 * theta * (1.0 + rho * phi * k + root)


def ssvi_derivatives(
    log_moneyness: ArrayLike,
    theta: float,
    rho: float,
    eta: float,
    gamma: float,
) -> tuple[FloatArray, FloatArray]:
    """Return analytic first and second log-moneyness derivatives."""
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be strictly between -1 and 1")
    k = np.asarray(log_moneyness, dtype=float)
    if np.any(~np.isfinite(k)):
        raise ValueError("log_moneyness must be finite")
    phi = float(power_law_phi(np.asarray([theta]), eta, gamma)[0])
    shifted = phi * k + rho
    root = np.sqrt(shifted**2 + 1.0 - rho**2)
    first = 0.5 * theta * phi * (rho + shifted / root)
    second = 0.5 * theta * phi**2 * (1.0 - rho**2) / root**3
    return first, second


def calibrate_ssvi(
    quotes: pd.DataFrame,
    *,
    diagnostic_grid: ArrayLike | None = None,
    max_iterations: int = 2000,
) -> SSVICalibration:
    """Jointly fit power-law SSVI under Gatheral--Jacquier conditions."""
    required = {"expiration", "maturity", "log_moneyness", "total_variance", "quote_count"}
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"surface quotes are missing: {', '.join(sorted(missing))}")
    if quotes["expiration"].nunique() < 2:
        raise ValueError("SSVI calibration requires at least two expirations")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    ordered = quotes.sort_values(["maturity", "log_moneyness"], ignore_index=True)
    k = ordered["log_moneyness"].to_numpy(dtype=float)
    observed = ordered["total_variance"].to_numpy(dtype=float)
    weights = ordered["quote_count"].to_numpy(dtype=float)
    if np.any(~np.isfinite(k)) or np.any(~np.isfinite(observed)) or np.any(observed <= 0):
        raise ValueError("SSVI observations must be finite with positive total variance")
    weights = weights / np.mean(weights)

    maturity_table = (
        ordered.groupby("expiration", as_index=False, sort=False)
        .agg(maturity=("maturity", "median"))
        .sort_values("maturity", ignore_index=True)
    )
    maturities = maturity_table["maturity"].to_numpy(dtype=float)
    expirations = tuple(str(value) for value in maturity_table["expiration"])
    maturity_index = np.searchsorted(maturities, ordered["maturity"].to_numpy(dtype=float))
    if np.any(maturity_index >= len(maturities)):
        raise ValueError("could not align SSVI observations to maturity nodes")
    if diagnostic_grid is None:
        largest = max(2.0, float(np.max(np.abs(k))) + 0.5)
        grid = np.linspace(-largest, largest, 161)
    else:
        grid = np.asarray(diagnostic_grid, dtype=float)
        if grid.ndim != 1 or grid.size < 21 or np.any(~np.isfinite(grid)):
            raise ValueError("diagnostic_grid must contain at least 21 finite points")

    theta_initial: list[float] = []
    for expiration in maturity_table["expiration"]:
        group = ordered.loc[ordered["expiration"] == expiration]
        closest = group.iloc[int(np.argmin(np.abs(group["log_moneyness"].to_numpy())))]
        theta_initial.append(float(closest["total_variance"]))
    theta_start = np.maximum.accumulate(np.asarray(theta_initial, dtype=float))
    for index in range(1, len(theta_start)):
        theta_start[index] = max(theta_start[index], theta_start[index - 1] + 1e-8)
    scale = max(float(np.median(observed)), 1e-6)

    def unpack(values: FloatArray) -> tuple[float, float, float, FloatArray]:
        return float(values[0]), float(values[1]), float(values[2]), values[3:]

    def objective(values: FloatArray) -> float:
        rho, eta, gamma, theta_nodes = unpack(values)
        fitted = np.empty_like(observed)
        for index, theta in enumerate(theta_nodes):
            mask = maturity_index == index
            fitted[mask] = ssvi_total_variance(k[mask], theta, rho, eta, gamma)
        residual = (fitted - observed) / scale
        return float(np.mean(weights * residual**2))

    theorem_margin = 1e-6

    def constraints(values: FloatArray) -> FloatArray:
        rho, eta, gamma, theta_nodes = unpack(values)
        phi = power_law_phi(theta_nodes, eta, gamma)
        correlation_scale = 1.0 + abs(rho)
        monotonic = np.diff(theta_nodes) - 1e-10
        butterfly_one = 4.0 - theorem_margin - theta_nodes * phi * correlation_scale
        butterfly_two = 4.0 - theta_nodes * phi**2 * correlation_scale
        return np.concatenate([monotonic, butterfly_one, butterfly_two])

    upper_theta = max(0.5, float(np.max(observed)) * 8.0)
    bounds = [(-0.999, 0.999), (1e-4, 5.0), (0.01, 0.99)] + [(1e-7, upper_theta)] * len(maturities)
    starts: list[FloatArray] = []
    for rho in (-0.7, -0.35, 0.0, 0.35):
        for eta in (0.2, 0.5, 1.0):
            starts.append(np.asarray([rho, eta, 0.5, *theta_start], dtype=float))

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
        feasible = float(np.min(constraints(np.asarray(result.x, dtype=float)))) >= -2e-6
        rank = (0 if result.success and feasible else 1 if feasible else 2, float(result.fun))
        if rank < best_rank:
            best = result
            best_rank = rank
    if best is None:
        raise RuntimeError("SSVI optimizer produced no result")

    rho, eta, gamma, theta_nodes = unpack(np.asarray(best.x, dtype=float))
    parameters = SSVIParameters(
        rho=rho,
        eta=eta,
        gamma=gamma,
        maturities=tuple(float(value) for value in maturities),
        theta=tuple(float(value) for value in theta_nodes),
    )
    fitted = np.empty_like(observed)
    reports: list[ButterflyReport] = []
    for index, theta in enumerate(theta_nodes):
        mask = maturity_index == index
        fitted[mask] = ssvi_total_variance(k[mask], theta, rho, eta, gamma)
        grid_variance = ssvi_total_variance(grid, theta, rho, eta, gamma)
        first, second = ssvi_derivatives(grid, theta, rho, eta, gamma)
        phi = float(power_law_phi(np.asarray([theta]), eta, gamma)[0])
        reports.append(
            butterfly_report(
                grid,
                grid_variance,
                first,
                second,
                left_wing_slope=0.5 * theta * phi * (1.0 - rho),
                right_wing_slope=0.5 * theta * phi * (1.0 + rho),
                tolerance=2e-6,
            )
        )

    def evaluator(maturity: float, values: FloatArray) -> FloatArray:
        return ssvi_total_variance(values, parameters.theta_at(maturity), rho, eta, gamma)

    calendar = calendar_report(maturities, evaluator, grid, tolerance=2e-6)
    constraint_minimum = float(np.min(constraints(np.asarray(best.x, dtype=float))))
    theorem_valid = constraint_minimum >= -2e-6
    optimizer_converged = bool(best.success and theorem_valid)
    observed_iv = np.sqrt(observed / ordered["maturity"].to_numpy(dtype=float))
    fitted_iv = np.sqrt(np.maximum(fitted, 0.0) / ordered["maturity"].to_numpy(dtype=float))
    reasons: list[str] = []
    if not optimizer_converged:
        reasons.append(f"SSVI optimizer failed: {best.message}")
    if not theorem_valid:
        reasons.append("SSVI Gatheral-Jacquier parameter conditions failed")
    if not all(report.is_arbitrage_free for report in reports):
        reasons.append("SSVI sampled butterfly diagnostic failed")
    if not calendar.is_arbitrage_free:
        reasons.append("SSVI sampled calendar-spread diagnostic failed")
    return SSVICalibration(
        parameters=parameters,
        expirations=expirations,
        optimizer_converged=optimizer_converged,
        iterations=int(best.nit),
        objective=float(best.fun),
        rmse_total_variance=float(np.sqrt(np.mean((fitted - observed) ** 2))),
        rmse_implied_volatility=float(np.sqrt(np.mean((fitted_iv - observed_iv) ** 2))),
        quote_count=len(observed),
        message=str(best.message),
        butterflies=tuple(reports),
        calendar=calendar,
        theorem_conditions_satisfied=theorem_valid,
        downstream_eligible=not reasons,
        failure_reasons=tuple(reasons),
    )
