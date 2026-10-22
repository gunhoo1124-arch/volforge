import json

import pytest

from volforge.deliverables import (
    parse_deliverable_terms,
    parse_security_marks,
    value_option_settlement,
)


def test_standard_physical_call_uses_multiplier_for_exercise_price() -> None:
    terms = parse_deliverable_terms(
        '[{"asset_id":"ABC","asset_type":"security","quantity":100}]',
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ABC",
    )
    result = value_option_settlement(
        terms,
        option_type="call",
        strike=100,
        underlying_mark=102,
    )

    assert terms.classification == "standard_physical"
    assert terms.single_factor_pricing_compatible
    assert result.delivered_value == pytest.approx(10_200)
    assert result.aggregate_exercise_price == pytest.approx(10_000)
    assert result.gross_receivable == pytest.approx(10_200)
    assert result.gross_payable == pytest.approx(10_000)
    assert result.net_payoff == pytest.approx(200)


def test_adjusted_physical_basket_values_security_cash_and_cash_in_lieu() -> None:
    terms = parse_deliverable_terms(
        [
            {"asset_id": "ACME", "asset_type": "security", "quantity": 50},
            {"asset_id": "SPIN", "asset_type": "security", "quantity": 10},
            {"asset_id": "USD", "asset_type": "cash", "quantity": 15},
            {
                "asset_id": "USD-CIL",
                "asset_type": "cash_in_lieu",
                "quantity": 5,
            },
        ],
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ACME",
    )
    result = value_option_settlement(
        terms,
        option_type="call",
        strike=60,
        security_marks={"ACME": 120, "SPIN": 20},
    )

    assert terms.classification == "adjusted_physical"
    assert not terms.single_factor_pricing_compatible
    assert terms.required_security_marks == ("ACME", "SPIN")
    assert result.delivered_value == pytest.approx(6_220)
    assert result.aggregate_exercise_price == pytest.approx(6_000)
    assert result.net_payoff == pytest.approx(220)
    assert result.gross_receivable - result.gross_payable == pytest.approx(result.net_payoff)
    assert result.to_dict()["components"][0]["asset_type"] == "cash"


def test_physical_put_reports_delivery_and_exercise_cash_flows() -> None:
    terms = parse_deliverable_terms(
        '[{"asset_id":"ABC","asset_type":"security","quantity":100}]',
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ABC",
    )
    result = value_option_settlement(
        terms,
        option_type="put",
        strike=100,
        security_marks={"ABC": 97},
    )

    assert result.gross_receivable == pytest.approx(10_000)
    assert result.gross_payable == pytest.approx(9_700)
    assert result.net_payoff == pytest.approx(300)


def test_missing_and_unused_security_marks_fail_closed() -> None:
    terms = parse_deliverable_terms(
        [
            {"asset_id": "ABC", "asset_type": "security", "quantity": 50},
            {"asset_id": "SPIN", "asset_type": "security", "quantity": 10},
        ],
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ABC",
    )
    with pytest.raises(ValueError, match="missing security marks: SPIN"):
        value_option_settlement(
            terms,
            option_type="call",
            strike=60,
            security_marks={"ABC": 120},
        )
    with pytest.raises(ValueError, match="unused security marks: TYPO"):
        value_option_settlement(
            terms,
            option_type="call",
            strike=60,
            security_marks={"ABC": 120, "SPIN": 20, "TYPO": 1},
        )


def test_deliverable_hash_is_independent_of_component_order() -> None:
    components = [
        {"asset_id": "ABC", "asset_type": "security", "quantity": 50},
        {"asset_id": "USD", "asset_type": "cash", "quantity": 12.5},
    ]
    forward = parse_deliverable_terms(
        components,
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ABC",
    )
    reverse = parse_deliverable_terms(
        list(reversed(components)),
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ABC",
    )

    assert forward.canonical_json == reverse.canonical_json
    assert forward.sha256 == reverse.sha256


@pytest.mark.parametrize(
    ("option_type", "settlement_level", "expected"),
    [("call", 605.0, 500.0), ("put", 595.0, 500.0)],
)
def test_cash_settled_payoff_uses_explicit_level_and_multiplier(
    option_type: str,
    settlement_level: float,
    expected: float,
) -> None:
    terms = parse_deliverable_terms(
        '[{"asset_id":"SPX-SET","asset_type":"cash_settlement","quantity":100,"currency":"USD"}]',
        settlement_method="cash",
        contract_multiplier=100,
        currency="USD",
        underlying_id="SPX",
    )
    result = value_option_settlement(
        terms,
        option_type=option_type,
        strike=600,
        settlement_level=settlement_level,
    )

    assert terms.classification == "cash_settled"
    assert not terms.single_factor_pricing_compatible
    assert result.delivered_value == pytest.approx(settlement_level * 100)
    assert result.net_payoff == pytest.approx(expected)
    assert result.gross_receivable == pytest.approx(expected)
    assert result.gross_payable == 0.0


def test_invalid_deliverable_shapes_fail_closed() -> None:
    common = {
        "settlement_method": "physical",
        "contract_multiplier": 100,
        "currency": "USD",
        "underlying_id": "ABC",
    }
    with pytest.raises(ValueError, match="duplicate JSON field 'quantity'"):
        parse_deliverable_terms(
            '[{"asset_id":"ABC","asset_type":"security","quantity":100,"quantity":50}]',
            **common,
        )
    with pytest.raises(ValueError, match="asset_type must be one of"):
        parse_deliverable_terms(
            [{"asset_id": "UNKNOWN", "asset_type": "other", "quantity": 1}],
            **common,
        )
    with pytest.raises(ValueError, match="currency differs"):
        parse_deliverable_terms(
            [
                {
                    "asset_id": "EUR-CASH",
                    "asset_type": "cash",
                    "quantity": 10,
                    "currency": "EUR",
                }
            ],
            **common,
        )
    with pytest.raises(ValueError, match="physical settlement cannot contain"):
        parse_deliverable_terms(
            [{"asset_id": "ABC-SET", "asset_type": "cash_settlement", "quantity": 100}],
            **common,
        )


def test_marks_json_is_strict() -> None:
    assert parse_security_marks('{"ABC":102.5}') == {"ABC": 102.5}
    with pytest.raises(ValueError, match="duplicate JSON field 'ABC'"):
        parse_security_marks('{"ABC":101,"ABC":102}')
    with pytest.raises(ValueError, match="must be non-negative"):
        parse_security_marks(json.dumps({"ABC": -1}))
