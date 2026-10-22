import pytest

from volforge.american import (
    BinomialSettings,
    LSMSettings,
    american_binomial_price,
    heston_american_pde_price,
    heston_lsm_price,
)
from volforge.models.heston import HestonParameters
from volforge.numerics import PDEGridSettings

PAPER_PARAMETERS = HestonParameters(1.5768, 0.0398, 0.5751, -0.5711, 0.0175)


def test_crr_matches_published_american_put_benchmark() -> None:
    result = american_binomial_price(
        "put", 40.0, 40.0, 1.0, 0.2, 0.06, settings=BinomialSettings(1_000)
    )
    assert result.price == pytest.approx(2.3193, abs=2e-4)
    assert result.european_price == pytest.approx(2.0664, abs=2e-4)
    assert result.early_exercise_premium > 0.0
    assert result.exercise_boundary


def test_non_dividend_american_call_equals_european_value() -> None:
    tree = american_binomial_price(
        "call", 100.0, 100.0, 1.0, 0.2, 0.05, settings=BinomialSettings(1_000)
    )
    lsm = heston_lsm_price(
        "call",
        100.0,
        100.0,
        1.0,
        PAPER_PARAMETERS,
        0.03,
        settings=LSMSettings(paths=4_000, steps=24, seed=7),
    )
    assert tree.price == pytest.approx(tree.european_price, abs=2e-3)
    assert not tree.exercise_boundary
    assert lsm.price == pytest.approx(lsm.european_price, abs=1e-12)
    assert lsm.exercise_fraction == 0.0


def test_heston_american_engines_are_seeded_and_cross_check() -> None:
    lsm_settings = LSMSettings(paths=4_000, steps=32, seed=7)
    first = heston_lsm_price(
        "put", 100.0, 100.0, 1.0, PAPER_PARAMETERS, 0.03, settings=lsm_settings
    )
    second = heston_lsm_price(
        "put", 100.0, 100.0, 1.0, PAPER_PARAMETERS, 0.03, settings=lsm_settings
    )
    pde = heston_american_pde_price(
        "put",
        100.0,
        100.0,
        1.0,
        PAPER_PARAMETERS,
        0.03,
        settings=PDEGridSettings(log_price_points=61, variance_points=31, time_steps=60),
    )
    assert first == second
    assert first.price > first.european_price
    assert pde.price > pde.european_price
    assert first.price == pytest.approx(pde.price, abs=3.0 * first.standard_error)
    assert pde.exercise_boundary


def test_heston_obstacle_pde_matches_constant_volatility_limit() -> None:
    parameters = HestonParameters(3.0, 0.04, 0.01, 0.0, 0.04)
    tree = american_binomial_price(
        "put", 100.0, 100.0, 1.0, 0.2, 0.05, settings=BinomialSettings(1_000)
    )
    pde = heston_american_pde_price(
        "put",
        100.0,
        100.0,
        1.0,
        parameters,
        0.05,
        settings=PDEGridSettings(
            log_price_points=61,
            variance_points=31,
            time_steps=60,
            variance_max=0.3,
        ),
    )
    assert pde.price == pytest.approx(tree.price, abs=0.12)


def test_american_settings_and_expiry_validation() -> None:
    expiry = american_binomial_price("put", 90.0, 100.0, 0.0, 0.2)
    assert expiry.price == 10.0
    with pytest.raises(ValueError, match="positive integer"):
        BinomialSettings(0)
    with pytest.raises(ValueError, match="even path"):
        LSMSettings(paths=101)
