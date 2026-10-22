"""Robust implied-volatility inversion with structured diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.optimize import brentq

from volforge.models import bachelier, black_scholes
from volforge.models.common import OptionType

SolverMethod = Literal["brent", "newton"]
VolatilityModel = Literal["bsm", "bachelier"]


@dataclass(frozen=True, slots=True)
class ImpliedVolResult:
    """Outcome of an implied-volatility solve."""

    volatility: float | None
    converged: bool
    iterations: int
    residual: float | None
    method_used: str
    failure_reason: str | None = None


def _failed(reason: str, method: str, residual: float | None = None) -> ImpliedVolResult:
    return ImpliedVolResult(None, False, 0, residual, method, reason)


def implied_volatility(
    option_price: float,
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    model: VolatilityModel = "bsm",
    method: SolverMethod = "brent",
    initial_guess: float | None = None,
    price_tolerance: float = 1e-10,
    volatility_tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> ImpliedVolResult:
    """Invert a BSM or Bachelier European option price.

    Brent's bracketed method is the reliable default. ``method='newton'``
    attempts safeguarded Newton iterations before falling back to Brent.
    """
    if not isinstance(option_price, (int, float)) or not math.isfinite(float(option_price)):
        return _failed("option_price must be a finite number", method)
    if option_price < 0:
        return _failed("option_price must be non-negative", method)
    if price_tolerance <= 0 or volatility_tolerance <= 0 or max_iterations <= 0:
        return _failed("solver tolerances and max_iterations must be positive", method)
    if model not in ("bsm", "bachelier"):
        return _failed("model must be 'bsm' or 'bachelier'", method)
    if method not in ("brent", "newton"):
        return _failed("method must be 'brent' or 'newton'", method)

    module = black_scholes if model == "bsm" else bachelier
    try:
        kind = OptionType.parse(option_type)
        lower, upper = module.bounds(kind, spot, strike, maturity, rate, dividend_yield)
    except ValueError as exc:
        return _failed(str(exc), method)

    scale_tolerance = price_tolerance * max(1.0, abs(option_price), abs(lower), abs(upper))
    if option_price < lower - scale_tolerance:
        return _failed(
            f"price is below the no-arbitrage lower bound ({lower:.12g})",
            method,
            option_price - lower,
        )
    if option_price > upper + scale_tolerance:
        return _failed(
            f"price is above the no-arbitrage upper bound ({upper:.12g})",
            method,
            option_price - upper,
        )
    if maturity == 0:
        return _failed("implied volatility is undefined at expiry", method, option_price - lower)
    if abs(option_price - lower) <= scale_tolerance:
        residual = (
            module.price(kind, spot, strike, maturity, 0.0, rate, dividend_yield) - option_price
        )
        return ImpliedVolResult(0.0, True, 0, residual, "boundary")
    if abs(option_price - upper) <= scale_tolerance:
        return _failed(
            "price at the upper bound implies unbounded volatility", method, option_price - upper
        )

    def objective(volatility: float) -> float:
        return (
            module.price(kind, spot, strike, maturity, volatility, rate, dividend_yield)
            - option_price
        )

    def vega(volatility: float) -> float:
        return module.greeks(kind, spot, strike, maturity, volatility, rate, dividend_yield).vega

    low = 0.0
    high = 1.0 if model == "bsm" else max(spot, strike, 1.0) / math.sqrt(maturity)
    high_limit = 1e4 if model == "bsm" else max(spot, strike, 1.0) * 1e6
    while objective(high) < 0 and high < high_limit:
        high *= 2.0
    if objective(high) < 0:
        return _failed("could not bracket a finite implied volatility", method, objective(high))

    newton_iterations = 0
    if method == "newton":
        guess_default = 0.2 if model == "bsm" else max(spot, strike, 1.0) * 0.2
        current = min(max(initial_guess if initial_guess is not None else guess_default, low), high)
        for newton_iterations in range(1, min(max_iterations, 25) + 1):
            residual = objective(current)
            if abs(residual) <= scale_tolerance:
                return ImpliedVolResult(current, True, newton_iterations, residual, "newton")
            if residual > 0:
                high = current
            else:
                low = current
            slope = vega(current)
            if not math.isfinite(slope) or slope <= 1e-14:
                break
            candidate = current - residual / slope
            if not math.isfinite(candidate) or candidate <= low or candidate >= high:
                break
            current = candidate

    try:
        root, details = brentq(
            objective,
            low,
            high,
            xtol=volatility_tolerance,
            rtol=max(4.0 * math.ulp(1.0), volatility_tolerance),
            maxiter=max_iterations,
            full_output=True,
            disp=False,
        )
    except (RuntimeError, ValueError) as exc:
        return _failed(
            f"bracketed solver failed: {exc}", "brent (fallback)" if method == "newton" else "brent"
        )
    residual = objective(root)
    used = "brent (fallback)" if method == "newton" else "brent"
    iterations = newton_iterations + int(details.iterations)
    if not details.converged or abs(residual) > max(scale_tolerance * 10.0, 1e-9):
        return ImpliedVolResult(
            root, False, iterations, residual, used, "solver did not meet residual tolerance"
        )
    return ImpliedVolResult(root, True, iterations, residual, used)


def black_scholes_implied_volatility(*args: object, **kwargs: object) -> ImpliedVolResult:
    """Solve for lognormal volatility; see :func:`implied_volatility`."""
    kwargs["model"] = "bsm"
    return implied_volatility(*args, **kwargs)  # type: ignore[arg-type]


def bachelier_implied_volatility(*args: object, **kwargs: object) -> ImpliedVolResult:
    """Solve for normal volatility; see :func:`implied_volatility`."""
    kwargs["model"] = "bachelier"
    return implied_volatility(*args, **kwargs)  # type: ignore[arg-type]
