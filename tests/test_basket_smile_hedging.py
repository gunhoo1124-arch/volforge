from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from volforge.basket_covariance import materialize_selected_covariance_replay
from volforge.basket_estimation import generate_synthetic_smile_basket_inputs
from volforge.basket_inputs import SYNTHETIC_BASKET_ID, BasketInputStore
from volforge.basket_replay import SYNTHETIC_BASKET_DELIVERABLE, SYNTHETIC_BASKET_EXPIRY
from volforge.basket_risk_hedging import (
    BasketRiskHedgeSettings,
    run_basket_risk_hedge_replay,
    verify_basket_risk_hedge_hash_chain,
)
from volforge.deliverables import DeliverableTerms, parse_deliverable_terms

SOURCE = "volforge-synthetic-statistical-v1"


@pytest.fixture(scope="module")
def smile_frame(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    records, schedule = generate_synthetic_smile_basket_inputs()
    root = tmp_path_factory.mktemp("smile-basket")
    store = BasketInputStore.create(root / "basket.sqlite")
    store.ingest_frame(
        records,
        source_file_name="smile.csv",
        source=SOURCE,
        mode="initial",
        ingested_timestamp="2026-05-01T00:00:00Z",
    )
    return materialize_selected_covariance_replay(
        store,
        schedule,
        basket_id=SYNTHETIC_BASKET_ID,
        source=SOURCE,
    )


@pytest.fixture(scope="module")
def smile_terms() -> DeliverableTerms:
    return parse_deliverable_terms(
        SYNTHETIC_BASKET_DELIVERABLE,
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ACME",
    )


def _run(frame: pd.DataFrame, terms: DeliverableTerms):
    return run_basket_risk_hedge_replay(
        frame,
        terms,
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=BasketRiskHedgeSettings(
            paths=2_000,
            include_smile_risks=True,
            require_covariance_selection=True,
        ),
    )


def test_dynamic_smile_hedge_is_rank_complete_neutral_and_audited(
    smile_frame: pd.DataFrame,
    smile_terms: DeliverableTerms,
) -> None:
    first = _run(smile_frame, smile_terms)
    second = _run(smile_frame, smile_terms)

    assert first == second
    assert first.completed and first.downstream_eligible
    assert first.smile_risk_coverage_bound
    assert first.covariance_selection_provenance_bound
    assert first.risk_factor_ids == (
        "spot_scale_gamma",
        "parallel_vega",
        "smile_skew",
        "smile_curvature",
        "correlation_strength",
    )
    assert verify_basket_risk_hedge_hash_chain(first)
    assert len(first.instrument_frame()) == 20
    for snapshot in first.snapshots[:-1]:
        assert snapshot.solver_rank == 5
        assert snapshot.solver_minimum_singular_value is not None
        assert snapshot.solver_minimum_singular_value > 1e-8
        assert snapshot.solver_condition_number is not None
        assert snapshot.solver_condition_number < 1_000.0
        assert snapshot.target_forward_log_moneyness is not None
        assert abs(snapshot.residual_smile_skew_sensitivity) < 1e-8
        assert abs(snapshot.residual_smile_curvature_sensitivity) < 1e-8
        assert len({item.forward_log_moneyness for item in snapshot.hedge_instruments}) == 5

    tampered_snapshot = replace(
        first.snapshots[0],
        target_smile_skew_sensitivity=(first.snapshots[0].target_smile_skew_sensitivity + 1.0),
    )
    tampered = replace(first, snapshots=(tampered_snapshot, *first.snapshots[1:]))
    assert not verify_basket_risk_hedge_hash_chain(tampered)


def test_smile_hedge_fails_closed_on_count_rank_and_moneyness(
    smile_frame: pd.DataFrame,
    smile_terms: DeliverableTerms,
) -> None:
    three_instruments = smile_frame.copy()
    three_instruments["hedge_instruments_json"] = [
        json.dumps(json.loads(value)[:3]) for value in three_instruments["hedge_instruments_json"]
    ]
    wrong_count = _run(three_instruments, smile_terms)
    assert not wrong_count.downstream_eligible
    assert "exactly 5 hedge instruments" in wrong_count.failure_reasons[0]

    rank_blocked = run_basket_risk_hedge_replay(
        smile_frame,
        smile_terms,
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=BasketRiskHedgeSettings(
            paths=2_000,
            include_smile_risks=True,
            minimum_normalized_singular_value=0.01,
        ),
    )
    assert not rank_blocked.downstream_eligible
    assert "does not span 5 declared risks" in rank_blocked.failure_reasons[0]

    moneyness_blocked = run_basket_risk_hedge_replay(
        smile_frame,
        smile_terms,
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=BasketRiskHedgeSettings(
            paths=2_000,
            include_smile_risks=True,
            maximum_absolute_smile_log_moneyness=0.01,
        ),
    )
    assert not moneyness_blocked.downstream_eligible
    assert "maximum absolute smile log-moneyness" in moneyness_blocked.failure_reasons[0]


def test_smile_settings_validate_and_legacy_schedule_remains_distinct() -> None:
    with pytest.raises(ValueError, match="minimum_normalized_singular_value"):
        BasketRiskHedgeSettings(minimum_normalized_singular_value=1.1)
    records, schedule = generate_synthetic_smile_basket_inputs()
    assert len(records) > 100
    assert all(len(json.loads(value)) == 5 for value in schedule["hedge_instruments_json"])
