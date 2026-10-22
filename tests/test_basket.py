import math

import numpy as np
import pytest

from volforge.basket import (
    BasketMarket,
    BasketMonteCarloSettings,
    parse_basket_market,
    price_basket_option,
)
from volforge.deliverables import parse_deliverable_terms
from volforge.models import black_scholes


def _standard_terms():  # type: ignore[no-untyped-def]
    return parse_deliverable_terms(
        '[{"asset_id":"ABC","asset_type":"security","quantity":100}]',
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ABC",
    )


def _one_asset_market() -> BasketMarket:
    return parse_basket_market(
        {
            "assets": [
                {
                    "asset_id": "ABC",
                    "spot": 100,
                    "volatility": 0.2,
                    "dividend_yield": 0.01,
                }
            ],
            "correlation": [[1]],
        }
    )


def test_single_asset_basket_matches_black_scholes_and_delta() -> None:
    settings = BasketMonteCarloSettings(paths=100_000, seed=7)
    result = price_basket_option(
        _standard_terms(),
        _one_asset_market(),
        option_type="call",
        strike=100,
        maturity=1,
        rate=0.03,
        settings=settings,
    )
    expected = black_scholes.price_and_greeks("call", 100, 100, 1, 0.2, 0.03, 0.01)

    assert result.moment_matched_price == pytest.approx(100 * expected.price, abs=1e-10)
    assert result.price == pytest.approx(
        100 * expected.price, abs=max(0.02, 4 * result.standard_error)
    )
    assert result.deltas[0].delta == pytest.approx(
        100 * expected.greeks.delta,
        abs=max(0.05, 4 * result.deltas[0].standard_error),
    )
    assert result.standard_error < result.crude_standard_error
    assert result.variance_reduction_ratio > 1.0
    assert result.bounds_passed


def test_antithetic_standard_error_uses_independent_pair_means() -> None:
    settings = BasketMonteCarloSettings(
        paths=20,
        seed=31,
        antithetic=True,
        control_variate=False,
    )
    result = price_basket_option(
        _standard_terms(),
        _one_asset_market(),
        option_type="call",
        strike=100,
        maturity=1,
        rate=0.03,
        settings=settings,
    )

    base = np.random.default_rng(31).standard_normal(10)
    shocks = np.concatenate((base, -base))
    terminal = 100 * np.exp((0.03 - 0.01 - 0.5 * 0.2**2) + 0.2 * shocks)
    payoffs = math.exp(-0.03) * np.maximum(100 * terminal - 10_000, 0)
    pair_means = 0.5 * (payoffs[:10] + payoffs[10:])
    expected_standard_error = np.std(pair_means, ddof=1) / math.sqrt(10)

    assert result.independent_samples == 10
    assert result.crude_standard_error == pytest.approx(expected_standard_error)


def test_adjusted_basket_is_seeded_and_carries_auditable_inputs() -> None:
    terms = parse_deliverable_terms(
        [
            {"asset_id": "ACME", "asset_type": "security", "quantity": 50},
            {"asset_id": "SPIN", "asset_type": "security", "quantity": 10},
            {"asset_id": "USD-CIL", "asset_type": "cash_in_lieu", "quantity": 20},
        ],
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ACME",
    )
    market = parse_basket_market(
        {
            "assets": [
                {"asset_id": "ACME", "spot": 120, "volatility": 0.25, "dividend_yield": 0.01},
                {"asset_id": "SPIN", "spot": 20, "volatility": 0.35, "dividend_yield": 0.0},
            ],
            "correlation": [[1.0, 0.35], [0.35, 1.0]],
        }
    )
    settings = BasketMonteCarloSettings(paths=40_000, seed=19)

    first = price_basket_option(
        terms,
        market,
        option_type="call",
        strike=60,
        maturity=0.5,
        rate=0.03,
        settings=settings,
    )
    second = price_basket_option(
        terms,
        market,
        option_type="call",
        strike=60,
        maturity=0.5,
        rate=0.03,
        settings=settings,
    )

    assert first == second
    assert first.classification == "adjusted_physical"
    assert first.risk_factor_ids == ("ACME", "SPIN")
    assert [delta.asset_id for delta in first.deltas] == ["ACME", "SPIN"]
    assert first.deliverable_sha256 == terms.sha256
    assert first.market_sha256 == market.sha256
    assert first.lower_bound <= first.price <= first.upper_bound
    assert first.to_dict()["deltas"][0]["asset_id"] == "ACME"


def test_cash_settled_pre_expiry_price_uses_underlying_risk_factor() -> None:
    terms = parse_deliverable_terms(
        '[{"asset_id":"SPX-SET","asset_type":"cash_settlement","quantity":100}]',
        settlement_method="cash",
        contract_multiplier=100,
        currency="USD",
        underlying_id="SPX",
    )
    market = parse_basket_market(
        {
            "assets": [
                {"asset_id": "SPX", "spot": 600, "volatility": 0.18, "dividend_yield": 0.015}
            ],
            "correlation": [[1]],
        }
    )
    result = price_basket_option(
        terms,
        market,
        option_type="put",
        strike=600,
        maturity=0.75,
        rate=0.04,
        settings=BasketMonteCarloSettings(paths=80_000, seed=11),
    )
    expected = black_scholes.price_and_greeks("put", 600, 600, 0.75, 0.18, 0.04, 0.015)

    assert result.settlement_method == "cash"
    assert result.risk_factor_ids == ("SPX",)
    assert result.moment_matched_price == pytest.approx(100 * expected.price, abs=1e-9)
    assert result.price == pytest.approx(
        100 * expected.price, abs=max(0.05, 4 * result.standard_error)
    )


def test_fixed_cash_only_contract_is_priced_without_simulated_assets() -> None:
    terms = parse_deliverable_terms(
        [{"asset_id": "USD-MERGER", "asset_type": "cash", "quantity": 120}],
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="OLDCO",
    )
    market = parse_basket_market({"assets": [], "correlation": []})
    result = price_basket_option(
        terms,
        market,
        option_type="call",
        strike=1,
        maturity=1,
        rate=0.05,
    )

    expected = 20 * math.exp(-0.05)
    assert result.price == pytest.approx(expected)
    assert result.standard_error == 0.0
    assert result.risk_factor_ids == ()
    assert result.deltas == ()
    assert result.lower_bound == pytest.approx(expected)


def test_market_parser_canonicalizes_asset_and_matrix_order() -> None:
    forward = parse_basket_market(
        {
            "assets": [
                {"asset_id": "A", "spot": 100, "volatility": 0.2},
                {"asset_id": "B", "spot": 80, "volatility": 0.3},
            ],
            "correlation": [[1, 0.4], [0.4, 1]],
        }
    )
    reverse = parse_basket_market(
        {
            "assets": [
                {"asset_id": "B", "spot": 80, "volatility": 0.3},
                {"asset_id": "A", "spot": 100, "volatility": 0.2},
            ],
            "correlation": [[1, 0.4], [0.4, 1]],
        }
    )

    assert forward.canonical_json == reverse.canonical_json
    assert forward.sha256 == reverse.sha256


def test_invalid_correlation_and_market_coverage_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        parse_basket_market(
            {
                "assets": [
                    {"asset_id": "A", "spot": 100, "volatility": 0.2},
                    {"asset_id": "B", "spot": 100, "volatility": 0.2},
                    {"asset_id": "C", "spot": 100, "volatility": 0.2},
                ],
                "correlation": [[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]],
            }
        )

    terms = parse_deliverable_terms(
        [
            {"asset_id": "A", "asset_type": "security", "quantity": 50},
            {"asset_id": "B", "asset_type": "security", "quantity": 50},
        ],
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="A",
    )
    incomplete = parse_basket_market(
        {
            "assets": [{"asset_id": "A", "spot": 100, "volatility": 0.2}],
            "correlation": [[1]],
        }
    )
    with pytest.raises(ValueError, match="missing basket assets: B"):
        price_basket_option(
            terms,
            incomplete,
            option_type="call",
            strike=100,
            maturity=1,
        )


def test_settings_and_duplicate_json_fields_are_validated() -> None:
    with pytest.raises(ValueError, match="integer"):
        BasketMonteCarloSettings(paths=10.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="even"):
        BasketMonteCarloSettings(paths=999)
    with pytest.raises(ValueError, match="at least 4"):
        BasketMonteCarloSettings(paths=2)
    with pytest.raises(ValueError, match="rows must be JSON lists"):
        parse_basket_market(
            {"assets": [{"asset_id": "A", "spot": 100, "volatility": 0.2}], "correlation": [1]}
        )
    with pytest.raises(ValueError, match="duplicate JSON field 'spot'"):
        parse_basket_market(
            '{"assets":[{"asset_id":"A","spot":100,"spot":101,'
            '"volatility":0.2}],"correlation":[[1]]}'
        )
