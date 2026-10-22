import math

import pytest

from volforge.models import bachelier


def test_atm_bachelier_value() -> None:
    expected = 20.0 / math.sqrt(2.0 * math.pi)
    assert bachelier.price("call", 100.0, 100.0, 1.0, 20.0) == pytest.approx(expected)
    assert bachelier.price("put", 100.0, 100.0, 1.0, 20.0) == pytest.approx(expected)


def test_bachelier_call_put_parity_with_carry() -> None:
    spot, strike, maturity, rate, dividend = 102.0, 100.0, 0.6, 0.03, 0.01
    call = bachelier.price("call", spot, strike, maturity, 18.0, rate, dividend)
    put = bachelier.price("put", spot, strike, maturity, 18.0, rate, dividend)
    expected = spot * math.exp(-dividend * maturity) - strike * math.exp(-rate * maturity)
    assert call - put == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bachelier_greeks_match_finite_differences(option_type: str) -> None:
    spot, strike, maturity, volatility, rate, dividend = 103.0, 99.0, 0.9, 16.0, 0.025, 0.01
    analytic = bachelier.greeks(option_type, spot, strike, maturity, volatility, rate, dividend)

    def value(s: float, t: float, vol: float, r: float) -> float:
        return bachelier.price(option_type, s, strike, t, vol, r, dividend)

    h_spot = 1e-2
    h_other = 1e-5
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
    assert analytic.gamma == pytest.approx(gamma, rel=5e-5)
    assert analytic.vega == pytest.approx(vega, rel=2e-7)
    assert analytic.theta == pytest.approx(theta, rel=2e-7)
    assert analytic.rho == pytest.approx(rho, rel=2e-7)


def test_bachelier_expiry_and_zero_volatility() -> None:
    assert bachelier.price("call", 110.0, 100.0, 0.0, 20.0) == 10.0
    assert bachelier.price("put", 90.0, 100.0, 0.0, 20.0) == 10.0
    assert bachelier.price("call", 100.0, 90.0, 1.0, 0.0) == 10.0
