"""Shared quote preparation and downstream surface gating."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import pandas as pd


class SurfaceQuoteError(ValueError):
    """Raised when an option chain cannot support surface calibration."""


class SurfaceValidationError(RuntimeError):
    """Raised when a calibrated surface fails the downstream arbitrage gate."""


class DownstreamEligible(Protocol):
    """Protocol implemented by calibration results accepted by the gate."""

    downstream_eligible: bool
    failure_reasons: tuple[str, ...]


def prepare_surface_quotes(
    chain: pd.DataFrame,
    *,
    minimum_strikes_per_expiry: int = 5,
) -> pd.DataFrame:
    """Convert an enriched BSM chain into unique total-variance observations.

    Calls and puts at the same strike/expiry are aggregated by their median
    implied volatility. Forward log-moneyness is ``log(K/F)`` with
    ``F = S exp((r-q)T)``.
    """
    required = {
        "expiration",
        "strike",
        "underlying_spot",
        "risk_free_rate",
        "dividend_yield",
        "maturity_years",
        "implied_volatility",
        "iv_converged",
        "is_valid",
    }
    missing = required.difference(chain.columns)
    if missing:
        raise SurfaceQuoteError(f"chain is missing surface columns: {', '.join(sorted(missing))}")
    if minimum_strikes_per_expiry < 5:
        raise ValueError("minimum_strikes_per_expiry must be at least five")

    usable = chain.loc[chain["is_valid"] & chain["iv_converged"], list(required)].copy()
    numeric_columns = [
        "strike",
        "underlying_spot",
        "risk_free_rate",
        "dividend_yield",
        "maturity_years",
        "implied_volatility",
    ]
    for column in numeric_columns:
        usable[column] = pd.to_numeric(usable[column], errors="coerce")
    usable = usable.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_columns)
    usable = usable.loc[
        (usable["strike"] > 0)
        & (usable["underlying_spot"] > 0)
        & (usable["maturity_years"] > 0)
        & (usable["implied_volatility"] > 0)
    ]
    if usable.empty:
        raise SurfaceQuoteError("no valid Black-Scholes implied-volatility observations")

    usable["forward"] = usable["underlying_spot"] * np.exp(
        (usable["risk_free_rate"] - usable["dividend_yield"]) * usable["maturity_years"]
    )
    usable["log_moneyness"] = np.log(usable["strike"] / usable["forward"])
    usable["total_variance"] = usable["implied_volatility"] ** 2 * usable["maturity_years"]
    grouped = (
        usable.groupby(["expiration", "strike"], as_index=False, sort=True)
        .agg(
            maturity=("maturity_years", "median"),
            forward=("forward", "median"),
            implied_volatility=("implied_volatility", "median"),
            total_variance=("total_variance", "median"),
            quote_count=("implied_volatility", "size"),
        )
        .sort_values(["maturity", "strike"], ignore_index=True)
    )
    grouped["log_moneyness"] = np.log(grouped["strike"] / grouped["forward"])

    counts = grouped.groupby("expiration")["strike"].nunique()
    sparse = counts[counts < minimum_strikes_per_expiry]
    if not sparse.empty:
        details = ", ".join(f"{expiry}: {count}" for expiry, count in sparse.items())
        raise SurfaceQuoteError(
            f"each expiry needs at least {minimum_strikes_per_expiry} unique strikes; {details}"
        )
    if grouped["expiration"].nunique() < 2:
        raise SurfaceQuoteError("surface calibration requires at least two expirations")
    if any(not math.isfinite(float(value)) for value in grouped["total_variance"]):
        raise SurfaceQuoteError("surface quotes contain non-finite total variance")
    return grouped


def require_downstream_eligible[CalibrationT: DownstreamEligible](
    calibration: CalibrationT,
) -> CalibrationT:
    """Return a calibration only if it passes the full static-arbitrage gate."""
    if calibration.downstream_eligible:
        return calibration
    reasons = calibration.failure_reasons or ("surface failed an unspecified validation",)
    raise SurfaceValidationError(
        "surface is blocked from downstream stochastic-volatility calibration: "
        + "; ".join(reasons)
    )
