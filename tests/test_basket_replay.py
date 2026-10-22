from dataclasses import replace

import pandas as pd
import pytest

from volforge.basket_replay import (
    SYNTHETIC_BASKET_DELIVERABLE,
    SYNTHETIC_BASKET_EXPIRY,
    BasketReplaySchemaError,
    BasketReplaySettings,
    basket_replay_report_json,
    generate_synthetic_basket_replay,
    run_basket_hedge_replay,
    verify_basket_replay_hash_chain,
)
from volforge.deliverables import parse_deliverable_terms


def _terms():  # type: ignore[no-untyped-def]
    return parse_deliverable_terms(
        SYNTHETIC_BASKET_DELIVERABLE,
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ACME",
    )


def _run(frame: pd.DataFrame, *, component_cost_bps: float = 1.0):  # type: ignore[no-untyped-def]
    return run_basket_hedge_replay(
        frame,
        _terms(),
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=BasketReplaySettings(
            paths=4_000,
            seed=31,
            component_cost_bps=component_cost_bps,
        ),
    )


def test_synthetic_basket_replay_is_deterministic_and_reconciles() -> None:
    frame = generate_synthetic_basket_replay()
    duplicate = generate_synthetic_basket_replay()
    pd.testing.assert_frame_equal(frame, duplicate)

    first = _run(frame)
    second = _run(frame)

    assert first == second
    assert first.completed
    assert first.downstream_eligible
    assert [snapshot.action for snapshot in first.snapshots] == [
        "open",
        "rebalance",
        "rebalance",
        "settle",
    ]
    assert all(snapshot.no_lookahead_passed for snapshot in first.snapshots)
    assert all(snapshot.pricing_bounds_passed for snapshot in first.snapshots)
    assert first.total_pnl == pytest.approx(first.snapshots[-1].cash)
    assert first.maximum_absolute_accounting_gap is not None
    assert first.maximum_absolute_accounting_gap < 1e-10
    assert first.total_spread_cost > 0.0
    assert first.total_transaction_cost > 0.0
    assert first.total_turnover > 0.0
    assert all(component.target_position == 0.0 for component in first.snapshots[-1].components)
    assert verify_basket_replay_hash_chain(first)
    assert b'"downstream_eligible": true' in basket_replay_report_json(first)
    assert len(first.ledger_frame()) == 4
    assert len(first.component_frame()) == 8


def test_future_received_market_package_blocks_at_first_invalid_snapshot() -> None:
    frame = generate_synthetic_basket_replay()
    frame.loc[1, "received_timestamp"] = "2026-04-10T20:01:00+00:00"

    result = _run(frame)

    assert not result.completed
    assert not result.downstream_eligible
    assert result.total_pnl is None
    assert len(result.snapshots) == 1
    assert "received_timestamp <= formation_timestamp" in result.failure_reasons[0]
    assert verify_basket_replay_hash_chain(result)


def test_missing_component_execution_quote_blocks_replay() -> None:
    frame = generate_synthetic_basket_replay()
    frame.loc[1, "execution_json"] = '[{"asset_id":"ACME","bid":121.98,"ask":122.02}]'

    result = _run(frame)

    assert not result.completed
    assert len(result.snapshots) == 1
    assert "missing execution quotes: SPIN" in result.failure_reasons[0]


def test_expiry_quote_must_equal_explicit_settlement_payoff() -> None:
    frame = generate_synthetic_basket_replay()
    frame.loc[3, "option_bid"] += 1.0
    frame.loc[3, "option_ask"] += 1.0

    result = _run(frame)

    assert not result.completed
    assert len(result.snapshots) == 3
    assert "must equal the explicit terminal settlement payoff" in result.failure_reasons[0]


def test_audit_mutation_is_detected() -> None:
    result = _run(generate_synthetic_basket_replay())
    changed = replace(result.snapshots[1], cash=result.snapshots[1].cash + 1.0)
    tampered = replace(result, snapshots=(result.snapshots[0], changed, *result.snapshots[2:]))

    assert not verify_basket_replay_hash_chain(tampered)
    settings_tampered = replace(
        result,
        settings=replace(result.settings, component_cost_bps=99.0),
    )
    assert not verify_basket_replay_hash_chain(settings_tampered)
    assert not verify_basket_replay_hash_chain(replace(result, total_pnl=result.total_pnl + 1.0))


def test_component_transaction_costs_reduce_final_pnl() -> None:
    frame = generate_synthetic_basket_replay()
    zero_cost = _run(frame, component_cost_bps=0.0)
    with_cost = _run(frame, component_cost_bps=25.0)

    assert zero_cost.completed and with_cost.completed
    assert zero_cost.total_pnl is not None and with_cost.total_pnl is not None
    assert with_cost.total_transaction_cost > zero_cost.total_transaction_cost
    assert with_cost.total_pnl < zero_cost.total_pnl


def test_cash_settled_index_can_use_explicit_proxy_hedge_quotes() -> None:
    terms = parse_deliverable_terms(
        '[{"asset_id":"SPX-SET","asset_type":"cash_settlement","quantity":100}]',
        settlement_method="cash",
        contract_multiplier=100,
        currency="USD",
        underlying_id="SPX",
    )
    frame = pd.DataFrame(
        [
            {
                "formation_timestamp": "2026-04-01T20:00:00+00:00",
                "market_timestamp": "2026-04-01T19:59:00+00:00",
                "received_timestamp": "2026-04-01T19:59:30+00:00",
                "source": "test",
                "source_record_id": "SPX-1",
                "option_bid": 450.0,
                "option_ask": 460.0,
                "risk_free_rate": 0.03,
                "market_json": (
                    '{"assets":[{"asset_id":"SPX","spot":600,"volatility":0.2,'
                    '"dividend_yield":0.01}],"correlation":[[1]]}'
                ),
                "execution_json": '[{"asset_id":"SPX","bid":599.9,"ask":600.1}]',
            },
            {
                "formation_timestamp": "2026-04-30T20:00:00+00:00",
                "market_timestamp": "2026-04-30T19:59:00+00:00",
                "received_timestamp": "2026-04-30T19:59:30+00:00",
                "source": "test",
                "source_record_id": "SPX-2",
                "option_bid": 1_000.0,
                "option_ask": 1_000.0,
                "risk_free_rate": 0.03,
                "market_json": (
                    '{"assets":[{"asset_id":"SPX","spot":610,"volatility":0.2,'
                    '"dividend_yield":0.01}],"correlation":[[1]]}'
                ),
                "execution_json": '[{"asset_id":"SPX","bid":609.9,"ask":610.1}]',
            },
        ]
    )

    result = run_basket_hedge_replay(
        frame,
        terms,
        option_type="call",
        strike=600,
        expiry_timestamp="2026-04-30T20:00:00+00:00",
        settings=BasketReplaySettings(paths=4_000),
    )

    assert result.completed
    assert result.snapshots[-1].settlement_payoff == pytest.approx(1_000.0)
    assert result.snapshots[0].components[0].asset_id == "SPX"


def test_replay_schema_and_settings_fail_closed() -> None:
    frame = generate_synthetic_basket_replay()
    with pytest.raises(BasketReplaySchemaError, match="missing basket replay columns"):
        _run(frame.drop(columns=["received_timestamp"]))
    with pytest.raises(BasketReplaySchemaError, match="strictly increasing"):
        _run(frame.iloc[[1, 0, 2, 3]].reset_index(drop=True))
    with pytest.raises(ValueError, match="transaction costs"):
        BasketReplaySettings(component_cost_bps=-1)
