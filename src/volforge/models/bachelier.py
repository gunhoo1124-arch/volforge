"""Bachelier (normal) European option pricing."""

from __future__ import annotations

import math

from scipy.stats import norm

from volforge.models.common import (
    Greeks,
    OptionType,
    PricingResult,
    discounted_price_bounds,
    expiry_delta,
    validate_inputs,
)

_DEGENERATE_VOL = 1e-12


def bounds(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return European bounds under the same continuous-carry convention."""
    return discounted_price_bounds(option_type, spot, strike, maturity, rate, dividend_yield)


def price(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    normal_volatility: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> float:
    """Price under Bachelier using normal volatility in price units/sqrt(year)."""
    validate_inputs(spot, strike, maturity, normal_volatility, rate, dividend_yield)
    kind = OptionType.parse(option_type)
    if maturity == 0:
        return max(spot - strike, 0.0) if kind is OptionType.CALL else max(strike - spot, 0.0)

    discount = math.exp(-rate * maturity)
    forward = spot * math.exp((rate - dividend_yield) * maturity)
    forward_intrinsic = forward - strike
    std_dev = normal_volatility * math.sqrt(maturity)
    if std_dev <= _DEGENERATE_VOL:
        payoff = (
            max(forward_intrinsic, 0.0) if kind is OptionType.CALL else max(-forward_intrinsic, 0.0)
        )
        return discount * payoff

    d = forward_intrinsic / std_dev
    if kind is OptionType.CALL:
        undiscounted = forward_intrinsic * norm.cdf(d) + std_dev * norm.pdf(d)
    else:
        undiscounted = -forward_intrinsic * norm.cdf(-d) + std_dev * norm.pdf(d)
    return discount * undiscounted


def greeks(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    normal_volatility: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> Greeks:
    """Return analytic normal-model Greeks where they are finite."""
    validate_inputs(spot, strike, maturity, normal_volatility, rate, dividend_yield)
    kind = OptionType.parse(option_type)
    if maturity == 0:
        return Greeks(expiry_delta(kind, spot, strike), 0.0, 0.0, 0.0, 0.0)

    discount = math.exp(-rate * maturity)
    carry = math.exp((rate - dividend_yield) * maturity)
    discount_q = math.exp(-dividend_yield * maturity)
    forward = spot * carry
    std_dev = normal_volatility * math.sqrt(maturity)
    if std_dev <= _DEGENERATE_VOL:
        call_indicator = 1.0 if forward > strike else 0.0 if forward < strike else 0.5
        psi = call_indicator if kind is OptionType.CALL else call_indicator - 1.0
        model_price = price(kind, spot, strike, maturity, 0.0, rate, dividend_yield)
        delta = discount_q * psi
        theta = rate * model_price - discount * (rate - dividend_yield) * forward * psi
        rho = maturity * (discount * forward * psi - model_price)
        return Greeks(delta, 0.0, 0.0, theta, rho)

    d = (forward - strike) / std_dev
    density = norm.pdf(d)
    psi = norm.cdf(d) if kind is OptionType.CALL else norm.cdf(d) - 1.0
    model_price = price(kind, spot, strike, maturity, normal_volatility, rate, dividend_yield)
    delta = discount_q * psi
    gamma = discount * carry**2 * density / std_dev
    vega = discount * math.sqrt(maturity) * density
    theta = rate * model_price - discount * (
        (rate - dividend_yield) * forward * psi
        + normal_volatility * density / (2.0 * math.sqrt(maturity))
    )
    rho = maturity * (discount * forward * psi - model_price)
    return Greeks(delta, gamma, vega, theta, rho)


def price_and_greeks(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    normal_volatility: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> PricingResult:
    """Return the Bachelier price and analytic Greeks."""
    return PricingResult(
        price(option_type, spot, strike, maturity, normal_volatility, rate, dividend_yield),
        greeks(option_type, spot, strike, maturity, normal_volatility, rate, dividend_yield),
    )
