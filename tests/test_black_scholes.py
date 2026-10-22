import math

import pytest

from volforge.models import black_scholes


def test_known_black_scholes_prices() -> None:
    call = black_scholes.price("call", 100.0, 100.0, 1.0, 0.2, 0.05)
    put = black_scholes.price("put", 100.0, 100.0, 1.0, 0.2, 0.05)
    assert call == pytest.approx(10.4505835722, rel=1e-10)
    assert put == pytest.approx(5.5735260223, rel=1e-10)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_prices_respect_bounds(option_type: str) -> None:
    value = black_scholes.price(option_type, 105.0, 100.0, 0.7, 0.35, 0.03, 0.01)
    lower, upper = black_scholes.bounds(option_type, 105.0, 100.0, 0.7, 0.03, 0.01)
    assert lower <= value <= upper


def test_call_put_parity() -> None:
    spot, strike, maturity, rate, dividend, volatility = 112.0, 104.0, 1.4, 0.04, 0.015, 0.27
    call = black_scholes.price("call", spot, strike, maturity, volatility, rate, dividend)
    put = black_scholes.price("put", spot, strike, maturity, volatility, rate, dividend)
    expected = spot * math.exp(-dividend * maturity) - strike * math.exp(-rate * maturity)
    assert call - put == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_analytic_greeks_match_finite_differences(option_type: str) -> None:
    spot, strike, maturity, volatility, rate, dividend = 103.0, 97.0, 0.8, 0.24, 0.035, 0.012
    analytic = black_scholes.greeks(option_type, spot, strike, maturity, volatility, rate, dividend)

    def value(s: float, t: float, vol: float, r: float) -> float:
        return black_scholes.price(option_type, s, strike, t, vol, r, dividend)

    h_spot = 1e-2
    base = value(spot, maturity, volatility, rate)
    delta = (
        value(spot + h_spot, maturity, volatility, rate)
        - value(spot - h_spot, maturity, volatility, rate)
    ) / (2 * h_spot)
    gamma = (
        value(spot + h_spot, maturity, volatility, rate)
        - 2 * base
        + value(spot - h_spot, maturity, volatility, rate)
    ) / h_spot**2
    h_other = 1e-5
    vega = (
        value(spot, maturity, volatility + h_other, rate)
        - value(spot, maturity, volatility - h_other, rate)
    ) / (2 * h_other)
    theta = (
        value(spot, maturity - h_other, volatility, rate)
        - value(spot, maturity + h_other, volatility, rate)
    ) / (2 * h_other)
    rho = (
        value(spot, maturity, volatility, rate + h_other)
        - value(spot, maturity, volatility, rate - h_other)
    ) / (2 * h_other)

    assert analytic.delta == pytest.approx(delta, rel=2e-7)
    assert analytic.gamma == pytest.approx(gamma, rel=2e-5)
    assert analytic.vega == pytest.approx(vega, rel=2e-7)
    assert analytic.theta == pytest.approx(theta, rel=2e-7)
    assert analytic.rho == pytest.approx(rho, rel=2e-7)


def test_expiry_and_zero_volatility_are_safe() -> None:
    assert black_scholes.price("call", 110.0, 100.0, 0.0, 0.2) == 10.0
    assert black_scholes.price("put", 90.0, 100.0, 0.0, 0.2) == 10.0
    assert black_scholes.price("call", 100.0, 90.0, 1.0, 0.0, 0.02) == pytest.approx(
        100.0 - 90.0 * math.exp(-0.02)
    )
    result = black_scholes.price_and_greeks("call", 100.0, 100.0, 0.5, 1e-15, 0.0)
    assert math.isfinite(result.price)
    assert all(
        math.isfinite(value)
        for value in (
            result.greeks.delta,
            result.greeks.gamma,
            result.greeks.vega,
            result.greeks.theta,
            result.greeks.rho,
        )
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"spot": 0.0}, "spot"),
        ({"strike": -1.0}, "strike"),
        ({"maturity": -0.1}, "maturity"),
        ({"volatility": -0.1}, "volatility"),
    ],
)
def test_invalid_parameters_raise(kwargs: dict[str, float], message: str) -> None:
    parameters = {"spot": 100.0, "strike": 100.0, "maturity": 1.0, "volatility": 0.2}
    parameters.update(kwargs)
    with pytest.raises(ValueError, match=message):
        black_scholes.price("call", **parameters)


def test_invalid_option_type_raises() -> None:
    with pytest.raises(ValueError, match="option_type must be"):
        black_scholes.price("straddle", 100.0, 100.0, 1.0, 0.2)
