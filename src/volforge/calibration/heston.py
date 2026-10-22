"""Heston calibration against an arbitrage-gated total-variance surface."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult, least_squares

from volforge.implied_vol import implied_volatility
from volforge.models import black_scholes
from volforge.models.heston import COSSettings, HestonParameters, cos_prices
from volforge.surfaces import SSVICalibration, SVICalibration, require_downstream_eligible

FloatArray = NDArray[np.float64]
_DEFAULT_CALIBRATION_COS_SETTINGS = COSSettings(terms=192)


class CalibratableSurface(Protocol):
    """Surface behavior required by Heston calibration."""

    downstream_eligible: bool
    failure_reasons: tuple[str, ...]
    maturities: tuple[float, ...]

    def total_variance(self, maturity: float, log_moneyness: ArrayLike) -> FloatArray:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class HestonCalibrationPoint:
    """One target/model comparison in normalized forward coordinates."""

    maturity: float
    log_moneyness: float
    strike: float
    target_total_variance: float
    target_implied_volatility: float
    target_price: float
    model_price: float
    model_implied_volatility: float | None
    price_residual: float
    implied_volatility_residual: float | None
    implied_volatility_converged: bool


@dataclass(frozen=True, slots=True)
class HestonCalibration:
    """Structured outcome of Heston calibration to a validated surface."""

    parameters: HestonParameters
    surface_model: str
    optimizer_converged: bool
    calibration_eligible: bool
    function_evaluations: int
    cost: float
    rmse_price: float
    rmse_implied_volatility: float | None
    maximum_absolute_price_error: float
    points: tuple[HestonCalibrationPoint, ...]
    message: str
    feller_required: bool
    failure_reasons: tuple[str, ...]
    cos_settings: COSSettings

    def evaluated_frame(self) -> pd.DataFrame:
        """Return target and model values as a presentation-neutral table."""
        return pd.DataFrame(asdict(point) for point in self.points)


def _surface_name(surface: CalibratableSurface) -> str:
    if isinstance(surface, SVICalibration):
        return "svi"
    if isinstance(surface, SSVICalibration):
        return "ssvi"
    return type(surface).__name__


def _calibration_targets(
    surface: CalibratableSurface,
    maturities: FloatArray,
    log_moneyness: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    maturity_values: list[float] = []
    k_values: list[float] = []
    total_variances: list[float] = []
    implied_volatilities: list[float] = []
    prices: list[float] = []
    for maturity in maturities:
        total_variance = np.asarray(
            surface.total_variance(float(maturity), log_moneyness), dtype=float
        )
        if total_variance.shape != log_moneyness.shape:
            raise ValueError("surface returned an invalid total-variance shape")
        if np.any(~np.isfinite(total_variance)) or np.any(total_variance <= 0):
            raise ValueError("surface returned non-positive or non-finite total variance")
        implied_volatility = np.sqrt(total_variance / maturity)
        strikes = np.exp(log_moneyness)
        for k, strike, variance, volatility in zip(
            log_moneyness,
            strikes,
            total_variance,
            implied_volatility,
            strict=True,
        ):
            maturity_values.append(float(maturity))
            k_values.append(float(k))
            total_variances.append(float(variance))
            implied_volatilities.append(float(volatility))
            prices.append(
                black_scholes.price("call", 1.0, float(strike), float(maturity), float(volatility))
            )
    return (
        np.asarray(maturity_values),
        np.asarray(k_values),
        np.asarray(total_variances),
        np.asarray(implied_volatilities),
        np.asarray(prices),
    )


def calibrate_heston(
    surface: CalibratableSurface,
    *,
    log_moneyness: ArrayLike = (-0.4, -0.2, 0.0, 0.2, 0.4),
    maturities: ArrayLike | None = None,
    settings: COSSettings = _DEFAULT_CALIBRATION_COS_SETTINGS,
    require_feller: bool = False,
    max_function_evaluations: int = 350,
) -> HestonCalibration:
    """Calibrate Heston only after the v0.2 static-arbitrage gate passes.

    Calibration is performed in normalized forward coordinates: ``F=1``,
    ``K=exp(k)``, and zero carry. Price residuals are divided by BSM vega,
    producing a stable first-order implied-volatility objective.
    """
    require_downstream_eligible(surface)
    k_grid = np.asarray(log_moneyness, dtype=float)
    if k_grid.ndim != 1 or k_grid.size < 5 or np.any(~np.isfinite(k_grid)):
        raise ValueError("Heston calibration requires at least five finite log-moneyness points")
    if np.any(np.diff(k_grid) <= 0):
        raise ValueError("log_moneyness points must be strictly increasing")
    maturity_grid = (
        np.asarray(surface.maturities, dtype=float)
        if maturities is None
        else np.asarray(maturities, dtype=float)
    )
    if (
        maturity_grid.ndim != 1
        or maturity_grid.size < 2
        or np.any(~np.isfinite(maturity_grid))
        or np.any(maturity_grid <= 0)
        or np.any(np.diff(maturity_grid) <= 0)
    ):
        raise ValueError("Heston calibration requires increasing positive maturities")
    if max_function_evaluations <= 0:
        raise ValueError("max_function_evaluations must be positive")

    maturity_values, k_values, target_w, target_iv, target_prices = _calibration_targets(
        surface, maturity_grid, k_grid
    )
    strikes = np.exp(k_values)
    vegas = np.asarray(
        [
            black_scholes.greeks("call", 1.0, strike, maturity, volatility).vega
            for strike, maturity, volatility in zip(
                strikes, maturity_values, target_iv, strict=True
            )
        ]
    )
    residual_scales = np.maximum(vegas, 0.02)
    maturity_groups = [np.flatnonzero(maturity_values == maturity) for maturity in maturity_grid]

    def unpack(values: FloatArray) -> HestonParameters:
        kappa, theta, third, rho, initial_variance = (float(value) for value in values)
        vol_of_vol = math.sqrt(2.0 * kappa * theta) * third if require_feller else third
        return HestonParameters(kappa, theta, vol_of_vol, rho, initial_variance)

    def model_prices(values: FloatArray) -> FloatArray:
        parameters = unpack(values)
        prices = np.empty_like(target_prices)
        for maturity, indices in zip(maturity_grid, maturity_groups, strict=True):
            prices[indices] = cos_prices(
                "call",
                1.0,
                strikes[indices],
                float(maturity),
                parameters,
                settings=settings,
            )
        return prices

    def residuals(values: FloatArray) -> FloatArray:
        try:
            prices = model_prices(values)
        except (ArithmeticError, FloatingPointError, ValueError):
            return np.full_like(target_prices, 1e3)
        if np.any(~np.isfinite(prices)):
            return np.full_like(target_prices, 1e3)
        return (prices - target_prices) / residual_scales

    atm_index = int(np.argmin(np.abs(k_grid)))
    short_variance = float(target_w[atm_index] / maturity_grid[0])
    long_offset = (len(maturity_grid) - 1) * len(k_grid) + atm_index
    long_variance = float(target_w[long_offset] / maturity_grid[-1])
    left_index = 0
    right_index = len(k_grid) - 1
    skew = target_iv[left_index] - target_iv[right_index]
    rho_guess = -0.5 if skew >= 0 else 0.5
    theta_guess = min(max(long_variance, 0.005), 0.5)
    v0_guess = min(max(short_variance, 0.005), 0.5)
    parameter_starts = (
        (1.5, theta_guess, 0.45, rho_guess, v0_guess),
        (0.6, theta_guess, 0.25, rho_guess * 0.6, v0_guess),
        (3.0, theta_guess, 0.8, rho_guess * 1.3, v0_guess),
        (1.0, (theta_guess + v0_guess) / 2.0, 0.6, 0.0, v0_guess),
    )
    if require_feller:
        starts = [
            np.asarray(
                [
                    kappa,
                    theta,
                    min(max(vol_of_vol / math.sqrt(2.0 * kappa * theta), 0.05), 0.95),
                    rho,
                    initial_variance,
                ]
            )
            for kappa, theta, vol_of_vol, rho, initial_variance in parameter_starts
        ]
        lower = np.asarray([0.05, 1e-4, 0.01, -0.995, 1e-4])
        upper = np.asarray([10.0, 1.0, 0.999, 0.995, 1.0])
    else:
        starts = [np.asarray(start, dtype=float) for start in parameter_starts]
        lower = np.asarray([0.05, 1e-4, 0.01, -0.995, 1e-4])
        upper = np.asarray([10.0, 1.0, 5.0, 0.995, 1.0])

    best: OptimizeResult | None = None
    for start in starts:
        result = least_squares(
            residuals,
            np.minimum(np.maximum(start, lower + 1e-10), upper - 1e-10),
            bounds=(lower, upper),
            max_nfev=max_function_evaluations,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            x_scale="jac",
        )
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        raise RuntimeError("Heston optimizer produced no result")

    parameters = unpack(np.asarray(best.x, dtype=float))
    fitted_prices = model_prices(np.asarray(best.x, dtype=float))
    points: list[HestonCalibrationPoint] = []
    iv_residuals: list[float] = []
    failures: list[str] = []
    for maturity, k, strike, variance, volatility, target_price, model_price in zip(
        maturity_values,
        k_values,
        strikes,
        target_w,
        target_iv,
        target_prices,
        fitted_prices,
        strict=True,
    ):
        solve = implied_volatility(model_price, "call", 1.0, strike, maturity)
        model_iv = solve.volatility if solve.converged else None
        iv_residual = model_iv - volatility if model_iv is not None else None
        if iv_residual is not None:
            iv_residuals.append(iv_residual)
        points.append(
            HestonCalibrationPoint(
                maturity=float(maturity),
                log_moneyness=float(k),
                strike=float(strike),
                target_total_variance=float(variance),
                target_implied_volatility=float(volatility),
                target_price=float(target_price),
                model_price=float(model_price),
                model_implied_volatility=float(model_iv) if model_iv is not None else None,
                price_residual=float(model_price - target_price),
                implied_volatility_residual=(
                    float(iv_residual) if iv_residual is not None else None
                ),
                implied_volatility_converged=solve.converged,
            )
        )
    if not best.success:
        failures.append(f"Heston optimizer failed: {best.message}")
    if len(iv_residuals) != len(points):
        failures.append("one or more Heston prices did not yield a finite BSM implied volatility")
    if require_feller and not parameters.feller_satisfied:
        failures.append("required Heston Feller condition failed")
    price_residuals = fitted_prices - target_prices
    return HestonCalibration(
        parameters=parameters,
        surface_model=_surface_name(surface),
        optimizer_converged=bool(best.success),
        calibration_eligible=not failures,
        function_evaluations=int(best.nfev),
        cost=float(best.cost),
        rmse_price=float(np.sqrt(np.mean(price_residuals**2))),
        rmse_implied_volatility=(
            float(np.sqrt(np.mean(np.asarray(iv_residuals) ** 2)))
            if len(iv_residuals) == len(points)
            else None
        ),
        maximum_absolute_price_error=float(np.max(np.abs(price_residuals))),
        points=tuple(points),
        message=str(best.message),
        feller_required=require_feller,
        failure_reasons=tuple(failures),
        cos_settings=settings,
    )


def heston_calibration_summary(calibration: HestonCalibration) -> dict[str, object]:
    """Return a deterministic JSON-compatible calibration audit."""
    return {
        "model": "heston",
        "source_surface": calibration.surface_model,
        "optimizer_converged": calibration.optimizer_converged,
        "calibration_eligible": calibration.calibration_eligible,
        "function_evaluations": calibration.function_evaluations,
        "cost": calibration.cost,
        "rmse_price": calibration.rmse_price,
        "rmse_implied_volatility": calibration.rmse_implied_volatility,
        "maximum_absolute_price_error": calibration.maximum_absolute_price_error,
        "quote_count": len(calibration.points),
        "message": calibration.message,
        "feller_required": calibration.feller_required,
        "feller_satisfied": calibration.parameters.feller_satisfied,
        "feller_margin": calibration.parameters.feller_margin,
        "failure_reasons": list(calibration.failure_reasons),
        "parameters": asdict(calibration.parameters),
        "cos_settings": asdict(calibration.cos_settings),
    }


def heston_calibration_report_json(calibration: HestonCalibration) -> bytes:
    """Serialize a deterministic human-readable Heston calibration audit."""
    return (
        json.dumps(heston_calibration_summary(calibration), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def heston_parameter_frame(calibration: HestonCalibration) -> pd.DataFrame:
    """Return one-row Heston parameters and diagnostics table."""
    return pd.DataFrame(
        [
            {
                **asdict(calibration.parameters),
                "feller_margin": calibration.parameters.feller_margin,
                "feller_satisfied": calibration.parameters.feller_satisfied,
                "iv_rmse": calibration.rmse_implied_volatility,
                "price_rmse": calibration.rmse_price,
                "optimizer_converged": calibration.optimizer_converged,
            }
        ]
    )
