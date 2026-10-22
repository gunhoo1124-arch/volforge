import pytest

from volforge.implied_vol import implied_volatility
from volforge.models import bachelier, black_scholes


@pytest.mark.parametrize(
    ("option_type", "spot", "strike", "maturity", "volatility"),
    [
        ("call", 100.0, 75.0, 0.25, 0.45),
        ("put", 100.0, 130.0, 0.40, 0.35),
        ("call", 100.0, 120.0, 2.0, 0.2),
        ("put", 100.0, 80.0, 1.0, 0.7),
    ],
)
def test_bsm_implied_volatility_round_trip(
    option_type: str, spot: float, strike: float, maturity: float, volatility: float
) -> None:
    option_price = black_scholes.price(option_type, spot, strike, maturity, volatility, 0.03, 0.01)
    result = implied_volatility(
        option_price, option_type, spot, strike, maturity, 0.03, 0.01, model="bsm"
    )
    assert result.converged, result.failure_reason
    assert result.volatility == pytest.approx(volatility, rel=2e-7, abs=1e-9)
    assert abs(result.residual or 0.0) < 1e-8


def test_bachelier_implied_volatility_round_trip() -> None:
    option_price = bachelier.price("put", 98.0, 105.0, 0.4, 17.5, 0.02, 0.01)
    result = implied_volatility(
        option_price, "put", 98.0, 105.0, 0.4, 0.02, 0.01, model="bachelier"
    )
    assert result.converged
    assert result.volatility == pytest.approx(17.5, rel=1e-9)


def test_newton_solver_or_safe_fallback() -> None:
    option_price = black_scholes.price("call", 100.0, 140.0, 0.05, 0.55)
    result = implied_volatility(
        option_price,
        "call",
        100.0,
        140.0,
        0.05,
        model="bsm",
        method="newton",
        initial_guess=0.01,
    )
    assert result.converged
    assert result.volatility == pytest.approx(0.55, rel=1e-7)
    assert result.method_used in {"newton", "brent (fallback)"}


def test_bound_violations_and_expiry_fail_explicitly() -> None:
    below = implied_volatility(-0.01, "call", 100.0, 100.0, 1.0)
    above = implied_volatility(101.0, "call", 100.0, 100.0, 1.0)
    expiry = implied_volatility(0.0, "call", 100.0, 100.0, 0.0)
    assert not below.converged and "non-negative" in (below.failure_reason or "")
    assert not above.converged and "upper bound" in (above.failure_reason or "")
    assert not expiry.converged and "undefined at expiry" in (expiry.failure_reason or "")


def test_lower_bound_maps_to_zero_volatility() -> None:
    lower, _ = black_scholes.bounds("call", 110.0, 100.0, 0.5, 0.02)
    result = implied_volatility(lower, "call", 110.0, 100.0, 0.5, 0.02)
    assert result.converged
    assert result.volatility == 0.0
    assert result.method_used == "boundary"


def test_unidentifiable_deep_short_dated_price_returns_boundary_safely() -> None:
    option_price = black_scholes.price("call", 100.0, 50.0, 0.01, 0.45, 0.03, 0.01)
    result = implied_volatility(option_price, "call", 100.0, 50.0, 0.01, 0.03, 0.01)
    assert result.converged
    assert result.volatility == 0.0
    assert result.method_used == "boundary"
