import numpy as np
import pytest

from volforge.models import black_scholes
from volforge.models.heston import (
    COSSettings,
    HestonParameters,
    cos_prices,
    heston_cumulants,
    heston_log_return_characteristic,
    price,
    price_with_diagnostics,
)

PAPER_PARAMETERS = HestonParameters(
    kappa=1.5768,
    theta=0.0398,
    vol_of_vol=0.5751,
    rho=-0.5711,
    initial_variance=0.0175,
)


def test_heston_characteristic_function_identity_and_symmetry() -> None:
    frequencies = np.asarray([0.0, 0.2, 1.0, 3.0])
    positive = heston_log_return_characteristic(frequencies, 1.0, PAPER_PARAMETERS)
    negative = heston_log_return_characteristic(-frequencies, 1.0, PAPER_PARAMETERS)
    assert positive[0] == pytest.approx(1.0 + 0.0j)
    assert negative == pytest.approx(np.conjugate(positive))


def test_heston_cos_matches_fang_oosterlee_benchmark() -> None:
    result = price(
        "call",
        100.0,
        100.0,
        1.0,
        PAPER_PARAMETERS,
        settings=COSSettings(terms=512),
    )
    assert result == pytest.approx(5.785155450, abs=2e-6)


def test_heston_cos_converges_to_black_scholes_when_variance_is_nearly_constant() -> None:
    volatility = 0.24
    parameters = HestonParameters(
        kappa=2.0,
        theta=volatility**2,
        vol_of_vol=1e-4,
        rho=0.0,
        initial_variance=volatility**2,
    )
    heston_price = price("put", 100.0, 105.0, 0.75, parameters, 0.03, 0.01)
    bsm_price = black_scholes.price("put", 100.0, 105.0, 0.75, volatility, 0.03, 0.01)
    assert heston_price == pytest.approx(bsm_price, abs=2e-5)


def test_heston_batch_prices_respect_parity_bounds_and_expiry() -> None:
    strikes = np.asarray([70.0, 90.0, 100.0, 110.0, 140.0])
    calls = cos_prices("call", 100.0, strikes, 1.25, PAPER_PARAMETERS, 0.03, 0.01)
    puts = cos_prices("put", 100.0, strikes, 1.25, PAPER_PARAMETERS, 0.03, 0.01)
    parity = 100.0 * np.exp(-0.01 * 1.25) - strikes * np.exp(-0.03 * 1.25)
    assert calls - puts == pytest.approx(parity, abs=1e-10)
    assert np.all(calls >= np.maximum(parity, 0.0))
    assert np.all(puts >= np.maximum(-parity, 0.0))
    assert cos_prices("call", 100.0, strikes, 0.0, PAPER_PARAMETERS) == pytest.approx(
        np.maximum(100.0 - strikes, 0.0)
    )


def test_heston_diagnostics_and_cumulants_are_finite() -> None:
    first, second = heston_cumulants(1.0, PAPER_PARAMETERS)
    result = price_with_diagnostics("call", 100.0, 100.0, 1.0, PAPER_PARAMETERS)
    assert np.isfinite([first, second]).all()
    assert second > 0.0
    assert result.lower_bound <= result.price <= result.upper_bound
    assert result.truncation_lower < result.truncation_upper
    assert not result.feller_satisfied
    assert result.feller_margin < 0.0


def test_heston_cumulants_match_characteristic_function_derivatives() -> None:
    first, second = heston_cumulants(1.0, PAPER_PARAMETERS, 0.03, 0.01)
    step = 1e-3
    log_positive = np.log(
        heston_log_return_characteristic([step], 1.0, PAPER_PARAMETERS, 0.03, 0.01)[0]
    )
    log_negative = np.log(
        heston_log_return_characteristic([-step], 1.0, PAPER_PARAMETERS, 0.03, 0.01)[0]
    )
    numerical_first = ((log_positive - log_negative) / (2j * step)).real
    numerical_second = -((log_positive + log_negative) / step**2).real
    assert first == pytest.approx(numerical_first, rel=5e-7)
    assert second == pytest.approx(numerical_second, rel=2e-6)


def test_heston_rejects_invalid_parameters_and_settings() -> None:
    with pytest.raises(ValueError, match="kappa"):
        HestonParameters(0.0, 0.04, 0.3, -0.5, 0.04)
    with pytest.raises(ValueError, match="rho"):
        HestonParameters(1.0, 0.04, 0.3, -1.0, 0.04)
    with pytest.raises(ValueError, match="at least 32"):
        COSSettings(terms=16)
    with pytest.raises(ValueError, match="strikes"):
        cos_prices("call", 100.0, [], 1.0, PAPER_PARAMETERS)
