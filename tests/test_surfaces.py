from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from volforge.data import enrich_option_chain, generate_synthetic_chain
from volforge.surfaces import (
    SurfaceQuoteError,
    SurfaceValidationError,
    SVIParameters,
    butterfly_report,
    calendar_report,
    calibrate_ssvi,
    calibrate_svi_slice,
    calibrate_svi_surface,
    prepare_surface_quotes,
    require_downstream_eligible,
    ssvi_derivatives,
    ssvi_total_variance,
    svi_derivatives,
    svi_total_variance,
)
from volforge.surfaces.arbitrage import density_factor
from volforge.surfaces.reporting import calibration_report_json, parameter_frame


def test_svi_derivatives_match_finite_differences() -> None:
    parameters = SVIParameters(a=0.015, b=0.08, rho=-0.35, m=0.02, sigma=0.22)
    k = np.asarray([-0.5, -0.1, 0.0, 0.3, 0.7])
    first, second = svi_derivatives(k, parameters)
    step = 1e-5
    center = svi_total_variance(k, parameters)
    plus = svi_total_variance(k + step, parameters)
    minus = svi_total_variance(k - step, parameters)
    assert first == pytest.approx((plus - minus) / (2 * step), rel=2e-8)
    assert second == pytest.approx((plus - 2 * center + minus) / step**2, rel=2e-5)


def test_ssvi_derivatives_match_finite_differences() -> None:
    k = np.asarray([-0.5, -0.1, 0.0, 0.3, 0.7])
    theta, rho, eta, gamma = 0.06, -0.4, 0.7, 0.5
    first, second = ssvi_derivatives(k, theta, rho, eta, gamma)
    step = 1e-5
    center = ssvi_total_variance(k, theta, rho, eta, gamma)
    plus = ssvi_total_variance(k + step, theta, rho, eta, gamma)
    minus = ssvi_total_variance(k - step, theta, rho, eta, gamma)
    assert first == pytest.approx((plus - minus) / (2 * step), rel=2e-8)
    assert second == pytest.approx((plus - 2 * center + minus) / step**2, rel=2e-5)


def test_density_factor_detects_butterfly_arbitrage() -> None:
    grid = np.linspace(-2.0, 2.0, 401)
    parameters = SVIParameters(a=-0.04, b=1.0, rho=0.9, m=0.0, sigma=0.1)
    variance = svi_total_variance(grid, parameters)
    first, second = svi_derivatives(grid, parameters)
    report = butterfly_report(
        grid,
        variance,
        first,
        second,
        left_wing_slope=parameters.left_wing_slope,
        right_wing_slope=parameters.right_wing_slope,
    )
    assert report.minimum_total_variance > 0
    assert report.minimum_density_factor < 0
    assert report.butterfly_violation_count > 0
    assert not report.is_arbitrage_free


def test_calendar_report_detects_crossing_total_variance() -> None:
    grid = np.linspace(-1.0, 1.0, 21)
    report = calendar_report([0.25, 0.5], lambda maturity, k: (1.0 - maturity) + 0.01 * k**2, grid)
    assert report.minimum_calendar_spread < 0
    assert report.violation_count == len(grid)
    assert not report.is_arbitrage_free


def test_constrained_svi_recovers_valid_synthetic_slice() -> None:
    grid = np.linspace(-0.7, 0.7, 21)
    expected = SVIParameters(a=0.018, b=0.06, rho=-0.4, m=0.03, sigma=0.25)
    observed = svi_total_variance(grid, expected)
    result = calibrate_svi_slice(grid, observed, 0.75, expiration="2030-06-01")
    assert result.downstream_eligible, result.message
    assert result.rmse_total_variance < 2e-6
    assert result.rmse_implied_volatility < 2e-5
    assert result.butterfly.minimum_density_factor >= -2e-6


def _exact_ssvi_quotes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rho, eta, gamma = -0.35, 0.6, 0.5
    for expiration, maturity, theta in (
        ("2030-03-01", 0.25, 0.012),
        ("2030-06-01", 0.5, 0.024),
        ("2031-01-01", 1.0, 0.05),
    ):
        for k in np.linspace(-0.6, 0.6, 13):
            total_variance = float(ssvi_total_variance([k], theta, rho, eta, gamma)[0])
            rows.append(
                {
                    "expiration": expiration,
                    "maturity": maturity,
                    "log_moneyness": k,
                    "total_variance": total_variance,
                    "implied_volatility": np.sqrt(total_variance / maturity),
                    "quote_count": 2,
                }
            )
    return pd.DataFrame(rows)


def test_ssvi_joint_calibration_is_static_arbitrage_free() -> None:
    result = calibrate_ssvi(_exact_ssvi_quotes(), diagnostic_grid=np.linspace(-1.5, 1.5, 121))
    assert result.downstream_eligible, result.failure_reasons
    assert result.theorem_conditions_satisfied
    assert result.calendar.is_arbitrage_free
    assert all(report.is_arbitrage_free for report in result.butterflies)
    assert result.rmse_total_variance < 2e-5
    assert require_downstream_eligible(result) is result
    assert b'"downstream_eligible": true' in calibration_report_json(result)
    assert len(parameter_frame(result)) == 3
    assert result.total_variance(0.375, [0.0])[0] > 0.0
    with pytest.raises(ValueError, match="limited to calibrated maturities"):
        result.total_variance(0.1, [0.0])


def test_default_chain_calibrates_constrained_svi_surface() -> None:
    chain = enrich_option_chain(generate_synthetic_chain())
    quotes = prepare_surface_quotes(chain)
    result = calibrate_svi_surface(quotes, diagnostic_grid=np.linspace(-1.25, 1.25, 121))
    assert len(quotes) == 28
    assert result.downstream_eligible, result.failure_reasons
    assert result.calendar.is_arbitrage_free
    assert len(result.slices) == 4
    evaluated = result.evaluated_frame(np.linspace(-0.5, 0.5, 11))
    assert len(evaluated) == 44
    assert (evaluated["total_variance"] > 0).all()


def test_quote_preparation_rejects_missing_or_sparse_data() -> None:
    with pytest.raises(SurfaceQuoteError, match="missing surface columns"):
        prepare_surface_quotes(pd.DataFrame({"strike": [100.0]}))

    chain = enrich_option_chain(generate_synthetic_chain())
    one_expiry = chain.loc[chain["expiration"] == chain["expiration"].iloc[0]]
    with pytest.raises(SurfaceQuoteError, match="at least two expirations"):
        prepare_surface_quotes(one_expiry)


def test_density_factor_rejects_nonpositive_variance() -> None:
    result = density_factor([0.0], [0.0], [0.0], [0.0])
    assert np.isneginf(result[0])


def test_downstream_gate_rejects_invalid_surface() -> None:
    @dataclass(frozen=True)
    class InvalidCalibration:
        downstream_eligible: bool = False
        failure_reasons: tuple[str, ...] = ("calendar-spread diagnostic failed",)

    with pytest.raises(SurfaceValidationError, match="calendar-spread diagnostic failed"):
        require_downstream_eligible(InvalidCalibration())
