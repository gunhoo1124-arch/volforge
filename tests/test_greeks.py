import pytest

from volforge.greeks import (
    FiniteDifferenceSettings,
    MonteCarloGreekSettings,
    heston_finite_difference_greeks,
    heston_monte_carlo_greeks,
)
from volforge.models import black_scholes
from volforge.models.heston import HestonParameters
from volforge.numerics import PDEGridSettings

PAPER_PARAMETERS = HestonParameters(1.5768, 0.0398, 0.5751, -0.5711, 0.0175)


def test_cos_greeks_match_black_scholes_constant_variance_limit() -> None:
    volatility = 0.24
    parameters = HestonParameters(2.0, volatility**2, 1e-4, 0.0, volatility**2)
    result = heston_finite_difference_greeks("call", 100.0, 105.0, 1.2, parameters, 0.03, 0.01)
    expected = black_scholes.price_and_greeks("call", 100.0, 105.0, 1.2, volatility, 0.03, 0.01)
    assert result.price == pytest.approx(expected.price, abs=1e-6)
    assert result.delta.value == pytest.approx(expected.greeks.delta, abs=2e-6)
    assert result.gamma.value == pytest.approx(expected.greeks.gamma, abs=2e-6)
    assert result.vega.value == pytest.approx(expected.greeks.vega, abs=2e-3)
    assert result.theta.value == pytest.approx(expected.greeks.theta, abs=2e-4)
    assert result.rho.value == pytest.approx(expected.greeks.rho, abs=2e-4)
    assert result.price_evaluations == 21


def test_monte_carlo_greeks_are_seeded_and_agree_with_cos() -> None:
    settings = MonteCarloGreekSettings(paths=10_000, steps=48, seed=7)
    first = heston_monte_carlo_greeks(
        "call", 100.0, 100.0, 1.0, PAPER_PARAMETERS, 0.03, settings=settings
    )
    second = heston_monte_carlo_greeks(
        "call", 100.0, 100.0, 1.0, PAPER_PARAMETERS, 0.03, settings=settings
    )
    reference = heston_finite_difference_greeks("call", 100.0, 100.0, 1.0, PAPER_PARAMETERS, 0.03)
    assert first == second
    assert first.price.confidence_low <= reference.price <= first.price.confidence_high
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        estimate = getattr(first, name)
        target = getattr(reference, name).value
        assert estimate.value == pytest.approx(target, abs=4.0 * estimate.standard_error + 0.02)
    assert first.delta_likelihood_ratio.value == pytest.approx(
        reference.delta.value,
        abs=4.0 * first.delta_likelihood_ratio.standard_error,
    )


def test_pde_greeks_cross_check_cos_on_a_fixed_grid() -> None:
    cos = heston_finite_difference_greeks("put", 100.0, 100.0, 1.0, PAPER_PARAMETERS, 0.03)
    pde = heston_finite_difference_greeks(
        "put",
        100.0,
        100.0,
        1.0,
        PAPER_PARAMETERS,
        0.03,
        engine="pde",
        pde_settings=PDEGridSettings(log_price_points=61, variance_points=31, time_steps=60),
    )
    assert pde.delta.value == pytest.approx(cos.delta.value, abs=0.03)
    assert pde.gamma.value == pytest.approx(cos.gamma.value, abs=0.02)
    assert pde.vega.value == pytest.approx(cos.vega.value, abs=5.0)
    assert pde.theta.value == pytest.approx(cos.theta.value, abs=0.4)
    assert pde.rho.value == pytest.approx(cos.rho.value, abs=2.0)


def test_greek_settings_and_contract_validation() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        FiniteDifferenceSettings(spot_relative_bump=0.0)
    with pytest.raises(ValueError, match="at least four"):
        MonteCarloGreekSettings(paths=2)
    with pytest.raises(ValueError, match="positive finite maturity"):
        heston_finite_difference_greeks("call", 100.0, 100.0, 0.0, PAPER_PARAMETERS)
    with pytest.raises(ValueError, match="must be 'cos' or 'pde'"):
        heston_finite_difference_greeks(
            "call", 100.0, 100.0, 1.0, PAPER_PARAMETERS, engine="unknown"
        )
