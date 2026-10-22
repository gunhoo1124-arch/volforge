import pytest

from volforge.models.heston import HestonParameters, price
from volforge.numerics import (
    ConvergenceSettings,
    MonteCarloSettings,
    NumericalConvergenceError,
    PDEGridSettings,
    assess_heston_convergence,
    convergence_report_json,
    heston_monte_carlo,
    heston_pde_price,
    require_numerical_convergence,
)

PAPER_PARAMETERS = HestonParameters(
    kappa=1.5768,
    theta=0.0398,
    vol_of_vol=0.5751,
    rho=-0.5711,
    initial_variance=0.0175,
)


def test_qe_monte_carlo_is_seeded_and_agrees_with_cos() -> None:
    settings = MonteCarloSettings(paths=40_000, steps=64, seed=7)
    first = heston_monte_carlo("call", 100.0, 100.0, 1.0, PAPER_PARAMETERS, settings=settings)
    second = heston_monte_carlo("call", 100.0, 100.0, 1.0, PAPER_PARAMETERS, settings=settings)
    reference = price("call", 100.0, 100.0, 1.0, PAPER_PARAMETERS)
    assert first == second
    assert first.price == pytest.approx(reference, abs=4.0 * first.standard_error + 0.02)
    assert first.confidence_low < first.price < first.confidence_high
    assert first.estimate_within_bounds
    assert first.lower_bound <= first.price <= first.upper_bound
    assert first.discounted_underlying_mean == pytest.approx(100.0, rel=2e-3)


def test_control_variate_reduces_monte_carlo_standard_error() -> None:
    plain = heston_monte_carlo(
        "put",
        100.0,
        105.0,
        1.0,
        PAPER_PARAMETERS,
        settings=MonteCarloSettings(
            paths=20_000,
            steps=48,
            seed=11,
            antithetic=False,
            control_variate=False,
        ),
    )
    controlled = heston_monte_carlo(
        "put",
        100.0,
        105.0,
        1.0,
        PAPER_PARAMETERS,
        settings=MonteCarloSettings(
            paths=20_000,
            steps=48,
            seed=11,
            antithetic=False,
            control_variate=True,
        ),
    )
    assert controlled.standard_error < plain.standard_error
    assert controlled.control_beta != 0.0


def test_pde_converges_toward_cos_and_preserves_put_call_parity() -> None:
    coarse = heston_pde_price(
        "call",
        100.0,
        100.0,
        1.0,
        PAPER_PARAMETERS,
        settings=PDEGridSettings(log_price_points=61, variance_points=31, time_steps=60),
    )
    settings = PDEGridSettings(log_price_points=121, variance_points=61, time_steps=180)
    call = heston_pde_price("call", 100.0, 100.0, 1.0, PAPER_PARAMETERS, settings=settings)
    put = heston_pde_price("put", 100.0, 100.0, 1.0, PAPER_PARAMETERS, settings=settings)
    reference = price("call", 100.0, 100.0, 1.0, PAPER_PARAMETERS)
    assert abs(call.price - reference) < abs(coarse.price - reference)
    assert call.price == pytest.approx(reference, abs=0.06)
    assert call.price - put.price == pytest.approx(0.0, abs=1e-10)
    assert call.log_price_spacing > 0.0
    assert call.variance_spacing > 0.0


def test_numerical_engines_handle_expiry_and_reject_invalid_settings() -> None:
    mc = heston_monte_carlo("put", 90.0, 100.0, 0.0, PAPER_PARAMETERS)
    pde = heston_pde_price("call", 110.0, 100.0, 0.0, PAPER_PARAMETERS)
    assert mc.price == 10.0 and mc.standard_error == 0.0
    assert pde.price == 10.0
    with pytest.raises(ValueError, match="even path"):
        MonteCarloSettings(paths=101)
    with pytest.raises(ValueError, match="initial variance"):
        heston_pde_price(
            "call",
            100.0,
            100.0,
            1.0,
            PAPER_PARAMETERS,
            settings=PDEGridSettings(variance_max=0.01),
        )


def test_automatic_convergence_gate_passes_and_is_deterministic() -> None:
    settings = ConvergenceSettings(
        absolute_tolerance=0.25,
        relative_tolerance=0.0,
        max_cos_terms=256,
        max_cos_truncation_width=16.0,
        max_monte_carlo_paths=20_000,
        max_monte_carlo_steps=64,
        max_pde_log_price_points=61,
        max_pde_variance_points=31,
        max_pde_time_steps=60,
        max_pde_log_price_half_width=4.0,
        max_pde_variance=1.0,
    )
    first = assess_heston_convergence(
        "call", 100.0, 100.0, 1.0, PAPER_PARAMETERS, settings=settings
    )
    second = assess_heston_convergence(
        "call", 100.0, 100.0, 1.0, PAPER_PARAMETERS, settings=settings
    )
    assert first == second
    assert first.downstream_eligible
    assert require_numerical_convergence(first) is first
    assert len(first.engines) == 3
    assert len(first.observations) == 9
    assert convergence_report_json(first) == convergence_report_json(second)


def test_automatic_convergence_gate_blocks_under_resolved_prices() -> None:
    settings = ConvergenceSettings(
        absolute_tolerance=1e-8,
        relative_tolerance=0.0,
        max_cos_terms=128,
        max_cos_truncation_width=10.0,
        max_monte_carlo_paths=4_000,
        max_monte_carlo_steps=16,
        max_pde_log_price_points=61,
        max_pde_variance_points=31,
        max_pde_time_steps=60,
        max_pde_log_price_half_width=2.0,
        max_pde_variance=0.5,
    )
    report = assess_heston_convergence(
        "call", 100.0, 100.0, 1.0, PAPER_PARAMETERS, settings=settings
    )
    assert not report.downstream_eligible
    assert report.failure_reasons
    with pytest.raises(NumericalConvergenceError, match="blocked from downstream"):
        require_numerical_convergence(report)


def test_convergence_settings_validate_budget_and_expiry_passes() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ConvergenceSettings(absolute_tolerance=0.0, relative_tolerance=0.0)
    report = assess_heston_convergence(
        "put",
        90.0,
        100.0,
        0.0,
        PAPER_PARAMETERS,
        settings=ConvergenceSettings(
            max_cos_terms=128,
            max_cos_truncation_width=10.0,
            max_monte_carlo_paths=4_000,
            max_monte_carlo_steps=16,
            max_pde_log_price_points=61,
            max_pde_variance_points=31,
            max_pde_time_steps=60,
            max_pde_log_price_half_width=2.0,
            max_pde_variance=0.5,
        ),
    )
    assert report.downstream_eligible
    assert report.reference_price == 10.0
