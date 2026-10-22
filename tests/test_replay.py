from dataclasses import replace

import pandas as pd
import pytest

from volforge.data import SyntheticReplayConfig, generate_synthetic_replay
from volforge.replay import (
    ReplaySettings,
    replay_report_json,
    run_chronological_replay,
    snapshot_sha256,
    verify_replay_hash_chain,
)


@pytest.fixture(scope="module")
def replay_chain() -> pd.DataFrame:
    return generate_synthetic_replay()


@pytest.fixture(scope="module")
def replay_result(replay_chain: pd.DataFrame):  # type: ignore[no-untyped-def]
    return run_chronological_replay(
        replay_chain,
        settings=ReplaySettings(heston_max_function_evaluations=60, cos_terms=64),
    )


def test_synthetic_replay_is_deterministic_and_keeps_fixed_contracts(
    replay_chain: pd.DataFrame,
) -> None:
    duplicate = generate_synthetic_replay(SyntheticReplayConfig())
    pd.testing.assert_frame_equal(replay_chain, duplicate)
    assert replay_chain["quote_timestamp"].nunique() == 4
    first_timestamp = replay_chain["quote_timestamp"].min()
    first = replay_chain.loc[replay_chain["quote_timestamp"] == first_timestamp]
    assert snapshot_sha256(first) == snapshot_sha256(first.sample(frac=1.0, random_state=13))


def test_chronological_replay_reconciles_and_verifies_hash_chain(replay_result) -> None:  # type: ignore[no-untyped-def]
    assert replay_result.completed
    assert replay_result.downstream_eligible
    assert verify_replay_hash_chain(replay_result)
    assert replay_result.target_contract is not None
    assert replay_result.hedge_contracts is not None
    assert [snapshot.action for snapshot in replay_result.snapshots] == [
        "open_rebalance",
        "rebalance",
        "rebalance",
        "liquidate",
    ]
    assert all(snapshot.no_lookahead_passed for snapshot in replay_result.snapshots)
    assert all(snapshot.surface_downstream_eligible for snapshot in replay_result.snapshots)
    assert all(snapshot.heston_calibration_eligible for snapshot in replay_result.snapshots)
    assert replay_result.maximum_absolute_accounting_gap is not None
    assert replay_result.maximum_absolute_accounting_gap < 1e-10
    assert replay_result.total_pnl == pytest.approx(replay_result.snapshots[-1].cash)
    assert b'"hash_chain_verified": true' in replay_report_json(replay_result)


def test_missing_later_listed_mark_blocks_without_contract_reselection(
    replay_chain: pd.DataFrame, replay_result
) -> None:  # type: ignore[no-untyped-def]
    high_hedge = replay_result.hedge_contracts[1]
    second_timestamp = sorted(replay_chain["quote_timestamp"].unique())[1]
    missing_mark = replay_chain.loc[
        ~(
            (replay_chain["quote_timestamp"] == second_timestamp)
            & (replay_chain["expiration"] == high_hedge.expiration)
            & (replay_chain["strike"] == high_hedge.strike)
            & (replay_chain["option_type"] == "call")
        )
    ]
    blocked = run_chronological_replay(
        missing_mark,
        settings=ReplaySettings(heston_max_function_evaluations=60, cos_terms=64),
    )
    assert not blocked.completed
    assert not blocked.downstream_eligible
    assert blocked.snapshots[-1].action == "blocked"
    assert blocked.target_contract == replay_result.target_contract
    assert blocked.hedge_contracts == replay_result.hedge_contracts
    assert any("exactly one valid quote" in reason for reason in blocked.failure_reasons)
    assert verify_replay_hash_chain(blocked)


def test_audit_mutation_is_detected(replay_result) -> None:  # type: ignore[no-untyped-def]
    changed_first = replace(replay_result.snapshots[0], cash=replay_result.snapshots[0].cash + 1.0)
    tampered = replace(
        replay_result,
        snapshots=(changed_first, *replay_result.snapshots[1:]),
    )
    assert not verify_replay_hash_chain(tampered)


def test_invalid_day_is_blocked_before_surface_or_heston(
    replay_chain: pd.DataFrame,
) -> None:
    invalid = replay_chain.copy()
    second_timestamp = sorted(invalid["quote_timestamp"].unique())[1]
    row_index = invalid.index[invalid["quote_timestamp"] == second_timestamp][0]
    invalid.loc[row_index, "bid"] = float(invalid.loc[row_index, "ask"]) + 1.0
    blocked = run_chronological_replay(
        invalid,
        settings=ReplaySettings(heston_max_function_evaluations=60, cos_terms=64),
    )
    failed_day = blocked.snapshots[-1]
    assert failed_day.action == "blocked"
    assert not failed_day.all_quotes_valid
    assert failed_day.surface_downstream_eligible is None
    assert failed_day.heston_calibration_eligible is None
    assert failed_day.heston_parameters is None


def test_explicit_reference_failure_cannot_reach_replay(
    replay_chain: pd.DataFrame,
) -> None:
    invalid = replay_chain.copy()
    invalid["reference_validated"] = True
    invalid["stable_contract_id"] = (
        invalid["expiration"].astype(str)
        + "-"
        + invalid["strike"].astype(str)
        + "-"
        + invalid["option_type"]
    )
    invalid.loc[invalid.index[0], "reference_validated"] = False
    with pytest.raises(ValueError, match="failed contract/session reference validation"):
        run_chronological_replay(invalid)


def test_options_execute_at_quote_sides_and_spread_is_reconciled(replay_result) -> None:  # type: ignore[no-untyped-def]
    first = replay_result.snapshots[0]
    last = replay_result.snapshots[-1]
    assert first.target_trade < 0.0
    assert first.target_fill == pytest.approx(first.target_bid)
    assert last.target_trade > 0.0
    assert last.target_fill == pytest.approx(last.target_ask)
    assert all(snapshot.spread_cost >= 0.0 for snapshot in replay_result.snapshots)
    assert replay_result.total_spread_cost == pytest.approx(
        sum(snapshot.spread_cost for snapshot in replay_result.snapshots)
    )
    assert replay_result.total_spread_cost > 0.0
    assert replay_result.snapshots[-1].cumulative_spread_cost == pytest.approx(
        replay_result.total_spread_cost
    )


def _standard_contract_terms(chain: pd.DataFrame, multiplier: float = 100.0) -> pd.DataFrame:
    enriched = chain.copy()
    enriched["stable_contract_id"] = (
        enriched["expiration"].astype(str)
        + "-"
        + enriched["strike"].astype(str)
        + "-"
        + enriched["option_type"]
    )
    enriched["contract_multiplier"] = multiplier
    enriched["deliverable_json"] = (
        '[{"asset_id":"SYNTHETIC","asset_type":"security","quantity":'
        + str(multiplier)
        + ',"currency":"USD"}]'
    )
    return enriched


def test_standard_contract_multiplier_scales_replay_economics(
    replay_chain: pd.DataFrame, replay_result
) -> None:  # type: ignore[no-untyped-def]
    scaled = run_chronological_replay(
        _standard_contract_terms(replay_chain),
        settings=ReplaySettings(heston_max_function_evaluations=60, cos_terms=64),
    )
    assert scaled.downstream_eligible
    assert scaled.total_pnl == pytest.approx(replay_result.total_pnl * 100.0)
    assert scaled.total_spread_cost == pytest.approx(replay_result.total_spread_cost * 100.0)
    assert scaled.total_transaction_cost == pytest.approx(
        replay_result.total_transaction_cost * 100.0
    )
    assert scaled.total_turnover == pytest.approx(replay_result.total_turnover * 100.0)
    assert scaled.snapshots[0].stock_position == pytest.approx(
        replay_result.snapshots[0].stock_position * 100.0
    )
    assert scaled.snapshots[0].target_multiplier == 100.0
    assert scaled.snapshots[0].target_deliverable_hash is not None


def test_structured_deliverable_blocks_before_surface_calibration(
    replay_chain: pd.DataFrame,
) -> None:
    adjusted = _standard_contract_terms(replay_chain)
    adjusted["deliverable_json"] = (
        '[{"asset_id":"SYNTHETIC","asset_type":"security","quantity":90},'
        '{"asset_id":"USD","asset_type":"cash","quantity":10}]'
    )
    blocked = run_chronological_replay(
        adjusted,
        settings=ReplaySettings(heston_max_function_evaluations=60, cos_terms=64),
    )
    assert not blocked.completed
    assert blocked.snapshots[0].surface_downstream_eligible is None
    assert any("unsupported structured deliverable" in reason for reason in blocked.failure_reasons)


def test_stable_contract_identity_prevents_same_terms_reselection(
    replay_chain: pd.DataFrame,
) -> None:
    referenced = _standard_contract_terms(replay_chain)
    first_timestamp, second_timestamp = sorted(referenced["quote_timestamp"].unique())[:2]
    first = referenced.loc[referenced["quote_timestamp"] == first_timestamp]
    first_expiration = sorted(first["expiration"].unique())[0]
    target_strike = min(first["strike"].unique(), key=lambda strike: abs(float(strike) - 100.0))
    replacement = (
        (referenced["quote_timestamp"] == second_timestamp)
        & (referenced["expiration"] == first_expiration)
        & (referenced["strike"] == target_strike)
        & (referenced["option_type"] == "call")
    )
    referenced.loc[replacement, "stable_contract_id"] = "replacement-contract"
    blocked = run_chronological_replay(
        referenced,
        settings=ReplaySettings(heston_max_function_evaluations=60, cos_terms=64),
    )
    assert not blocked.completed
    assert blocked.snapshots[-1].sequence == 1
    assert any("exactly one valid quote" in reason for reason in blocked.failure_reasons)
