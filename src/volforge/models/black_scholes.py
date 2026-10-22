"""Black--Scholes--Merton pricing with continuous carry."""

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
    """Return Black--Scholes European no-arbitrage bounds."""
    return discounted_price_bounds(option_type, spot, strike, maturity, rate, dividend_yield)


def price(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European call or put under Black--Scholes--Merton."""
    validate_inputs(spot, strike, maturity, volatility, rate, dividend_yield)
    kind = OptionType.parse(option_type)
    if maturity == 0:
        return max(spot - strike, 0.0) if kind is OptionType.CALL else max(strike - spot, 0.0)

    discounted_spot = spot * math.exp(-dividend_yield * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)
    if volatility <= _DEGENERATE_VOL:
        intrinsic_forward = discounted_spot - discounted_strike
        return (
            max(intrinsic_forward, 0.0) if kind is OptionType.CALL else max(-intrinsic_forward, 0.0)
        )

    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / (
        volatility * sqrt_t
    )
    d2 = d1 - volatility * sqrt_t
    if kind is OptionType.CALL:
        return discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    return discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)


def greeks(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> Greeks:
    """Return analytic BSM Greeks using annualized calendar-time theta."""
    validate_inputs(spot, strike, maturity, volatility, rate, dividend_yield)
    kind = OptionType.parse(option_type)
    if maturity == 0:
        return Greeks(expiry_delta(kind, spot, strike), 0.0, 0.0, 0.0, 0.0)

    discount_q = math.exp(-dividend_yield * maturity)
    discount_r = math.exp(-rate * maturity)
    forward_moneyness = spot * math.exp((rate - dividend_yield) * maturity) - strike
    if volatility <= _DEGENERATE_VOL:
        call_indicator = 1.0 if forward_moneyness > 0 else 0.0 if forward_moneyness < 0 else 0.5
        psi = call_indicator if kind is OptionType.CALL else call_indicator - 1.0
        model_price = price(kind, spot, strike, maturity, 0.0, rate, dividend_yield)
        delta = discount_q * psi
        rho = maturity * (spot * discount_q * psi - model_price)
        theta = rate * model_price - (rate - dividend_yield) * spot * discount_q * psi
        return Greeks(delta, 0.0, 0.0, theta, rho)

    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / (
        volatility * sqrt_t
    )
    d2 = d1 - volatility * sqrt_t
    density = norm.pdf(d1)
    gamma = discount_q * density / (spot * volatility * sqrt_t)
    vega = spot * discount_q * density * sqrt_t
    diffusion_decay = -(spot * discount_q * density * volatility) / (2.0 * sqrt_t)
    if kind is OptionType.CALL:
        delta = discount_q * norm.cdf(d1)
        theta = (
            diffusion_decay
            + dividend_yield * spot * discount_q * norm.cdf(d1)
            - rate * strike * discount_r * norm.cdf(d2)
        )
        rho = strike * maturity * discount_r * norm.cdf(d2)
    else:
        delta = discount_q * (norm.cdf(d1) - 1.0)
        theta = (
            diffusion_decay
            - dividend_yield * spot * discount_q * norm.cdf(-d1)
            + rate * strike * discount_r * norm.cdf(-d2)
        )
        rho = -strike * maturity * discount_r * norm.cdf(-d2)
    return Greeks(delta, gamma, vega, theta, rho)


def price_and_greeks(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> PricingResult:
    """Return the BSM price and analytic Greeks in one typed result."""
    return PricingResult(
        price(option_type, spot, strike, maturity, volatility, rate, dividend_yield),
        greeks(option_type, spot, strike, maturity, volatility, rate, dividend_yield),
    )
