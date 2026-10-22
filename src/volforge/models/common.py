"""Shared model types and validation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class OptionType(StrEnum):
    """Supported European option payoff types."""

    CALL = "call"
    PUT = "put"

    @classmethod
    def parse(cls, value: str | OptionType) -> OptionType:
        """Normalize a user-supplied option type."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError("option_type must be 'call' or 'put'") from exc


@dataclass(frozen=True, slots=True)
class Greeks:
    """Analytic sensitivities.

    Vega and rho are per absolute 1.00 change in volatility and rate. Divide
    them by 100 for a one-percentage-point move. Theta is calendar-time decay
    per year, i.e. minus the derivative with respect to time to maturity.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass(frozen=True, slots=True)
class PricingResult:
    """A model value and its analytic first/second-order sensitivities."""

    price: float
    greeks: Greeks


def validate_inputs(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float,
    dividend_yield: float,
) -> None:
    """Validate scalar European-option model inputs."""
    values = {
        "spot": spot,
        "strike": strike,
        "maturity": maturity,
        "volatility": volatility,
        "rate": rate,
        "dividend_yield": dividend_yield,
    }
    for name, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
    if spot <= 0:
        raise ValueError("spot must be greater than zero")
    if strike <= 0:
        raise ValueError("strike must be greater than zero")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")


def discounted_price_bounds(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return European no-arbitrage lower and upper price bounds."""
    validate_inputs(spot, strike, maturity, 0.0, rate, dividend_yield)
    kind = OptionType.parse(option_type)
    discounted_spot = spot * math.exp(-dividend_yield * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)
    if kind is OptionType.CALL:
        return max(discounted_spot - discounted_strike, 0.0), discounted_spot
    return max(discounted_strike - discounted_spot, 0.0), discounted_strike


def expiry_delta(option_type: OptionType, spot: float, strike: float) -> float:
    """Return the documented symmetric delta convention at expiry."""
    call_delta = 1.0 if spot > strike else 0.0 if spot < strike else 0.5
    return call_delta if option_type is OptionType.CALL else call_delta - 1.0
