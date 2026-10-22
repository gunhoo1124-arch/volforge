from dataclasses import dataclass

import numpy as np
import pytest

from volforge.calibration import calibrate_heston, heston_calibration_report_json
from volforge.implied_vol import implied_volatility
from volforge.models.heston import COSSettings, HestonParameters, cos_prices
from volforge.surfaces import SurfaceValidationError


@dataclass(frozen=True)
class ExactHestonSurface:
    parameters: HestonParameters
    maturities: tuple[float, ...] = (0.3, 0.8, 1.5)
    downstream_eligible: bool = True
    failure_reasons: tuple[str, ...] = ()

    def total_variance(self, maturity: float, log_moneyness: object) -> np.ndarray:
        k = np.asarray(log_moneyness, dtype=float)
        strikes = np.exp(k)
        prices = cos_prices(
            "call",
            1.0,
            strikes,
            maturity,
            self.parameters,
            settings=COSSettings(terms=256),
        )
        implied = [
            implied_volatility(option_price, "call", 1.0, strike, maturity).volatility
            for option_price, strike in zip(prices, strikes, strict=True)
        ]
        assert all(value is not None for value in implied)
        return np.asarray(implied, dtype=float) ** 2 * maturity


def test_heston_calibration_recovers_an_exact_gated_surface() -> None:
    expected = HestonParameters(1.5, 0.045, 0.42, -0.55, 0.032)
    result = calibrate_heston(
        ExactHestonSurface(expected),
        log_moneyness=np.linspace(-0.35, 0.35, 5),
        settings=COSSettings(terms=128),
        max_function_evaluations=180,
    )
    assert result.optimizer_converged, result.message
    assert result.calibration_eligible, result.failure_reasons
    assert result.rmse_implied_volatility is not None
    assert result.rmse_implied_volatility < 5e-4
    assert len(result.points) == 15
    assert b'"calibration_eligible": true' in heston_calibration_report_json(result)


def test_heston_calibration_can_require_the_feller_condition() -> None:
    expected = HestonParameters(2.0, 0.05, 0.35, -0.4, 0.04)
    result = calibrate_heston(
        ExactHestonSurface(expected),
        require_feller=True,
        settings=COSSettings(terms=96),
        max_function_evaluations=120,
    )
    assert result.parameters.feller_satisfied
    assert result.feller_required


def test_heston_calibration_refuses_an_ungated_surface() -> None:
    invalid = ExactHestonSurface(
        HestonParameters(1.5, 0.04, 0.4, -0.5, 0.04),
        downstream_eligible=False,
        failure_reasons=("calendar-spread diagnostic failed",),
    )
    with pytest.raises(SurfaceValidationError, match="calendar-spread diagnostic failed"):
        calibrate_heston(invalid)
