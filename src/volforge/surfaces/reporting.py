"""Deterministic surface-calibration tables and reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pandas as pd

from volforge.surfaces.ssvi import SSVICalibration
from volforge.surfaces.svi import SVICalibration

SurfaceCalibration = SVICalibration | SSVICalibration


def calibration_summary(calibration: SurfaceCalibration) -> dict[str, Any]:
    """Convert a surface result into a JSON-compatible audit summary."""
    base: dict[str, Any] = {
        "model": "svi" if isinstance(calibration, SVICalibration) else "ssvi",
        "downstream_eligible": calibration.downstream_eligible,
        "failure_reasons": list(calibration.failure_reasons),
        "rmse_total_variance": calibration.rmse_total_variance,
        "rmse_implied_volatility": calibration.rmse_implied_volatility,
        "calendar": asdict(calibration.calendar),
    }
    if isinstance(calibration, SVICalibration):
        base["slices"] = [
            {
                "expiration": result.expiration,
                "maturity": result.maturity,
                "optimizer_converged": result.optimizer_converged,
                "iterations": result.iterations,
                "objective": result.objective,
                "rmse_total_variance": result.rmse_total_variance,
                "rmse_implied_volatility": result.rmse_implied_volatility,
                "quote_count": result.quote_count,
                "message": result.message,
                "parameters": asdict(result.parameters),
                "butterfly": asdict(result.butterfly),
            }
            for result in calibration.slices
        ]
    else:
        base.update(
            {
                "optimizer_converged": calibration.optimizer_converged,
                "iterations": calibration.iterations,
                "objective": calibration.objective,
                "quote_count": calibration.quote_count,
                "message": calibration.message,
                "theorem_conditions_satisfied": calibration.theorem_conditions_satisfied,
                "parameters": asdict(calibration.parameters),
                "butterflies": [asdict(report) for report in calibration.butterflies],
                "expirations": list(calibration.expirations),
            }
        )
    return base


def calibration_report_json(calibration: SurfaceCalibration) -> bytes:
    """Serialize a deterministic, human-readable calibration report."""
    return (
        json.dumps(calibration_summary(calibration), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def parameter_frame(calibration: SurfaceCalibration) -> pd.DataFrame:
    """Return a presentation-neutral parameter table."""
    if isinstance(calibration, SVICalibration):
        return pd.DataFrame(
            [
                {
                    "expiration": result.expiration,
                    "maturity": result.maturity,
                    **asdict(result.parameters),
                    "iv_rmse": result.rmse_implied_volatility,
                    "minimum_g": result.butterfly.minimum_density_factor,
                    "butterfly_free": result.butterfly.is_arbitrage_free,
                    "optimizer_converged": result.optimizer_converged,
                }
                for result in calibration.slices
            ]
        )
    return pd.DataFrame(
        [
            {
                "expiration": expiration,
                "maturity": maturity,
                "theta": theta,
                "rho": calibration.parameters.rho,
                "eta": calibration.parameters.eta,
                "gamma": calibration.parameters.gamma,
                "minimum_g": report.minimum_density_factor,
                "butterfly_free": report.is_arbitrage_free,
                "optimizer_converged": calibration.optimizer_converged,
            }
            for expiration, maturity, theta, report in zip(
                calibration.expirations,
                calibration.parameters.maturities,
                calibration.parameters.theta,
                calibration.butterflies,
                strict=True,
            )
        ]
    )
