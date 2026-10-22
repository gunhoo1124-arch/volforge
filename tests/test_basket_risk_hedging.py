from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from volforge.basket_estimation import (
    generate_synthetic_statistical_basket_inputs,
    materialize_estimated_replay,
)
from volforge.basket_inputs import SYNTHETIC_BASKET_ID, BasketInputStore
from volforge.basket_replay import SYNTHETIC_BASKET_DELIVERABLE, SYNTHETIC_BASKET_EXPIRY
from volforge.basket_risk_hedging import (
    BasketRiskHedgeSchemaError,
    BasketRiskHedgeSettings,
    basket_risk_hedge_report_json,
    run_basket_risk_hedge_replay,
    verify_basket_risk_hedge_hash_chain,
)
from volforge.deliverables import DeliverableTerms, parse_deliverable_terms

SOURCE = "volforge-synthetic-statistical-v1"


@pytest.fixture(scope="module")
def estimated_frame(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    records, schedule = generate_synthetic_statistical_basket_inputs()
    root = tmp_path_factory.mktemp("advanced-basket")
    store = BasketInputStore.create(root / "basket.sqlite")
    store.ingest_frame(
        records,
        source_file_name="statistical.csv",
        source=SOURCE,
        mode="initial",
        ingested_timestamp="2026-05-01T00:00:00Z",
    )
    return materialize_estimated_replay(
        store,
        schedule,
        basket_id=SYNTHETIC_BASKET_ID,
        source=SOURCE,
    )


@pytest.fixture(scope="module")
def terms() -> DeliverableTerms:
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
        settings=BasketRiskHedgeSettings(paths=2_000),
    )


def test_advanced_hedge_is_deterministic_neutral_and_self_financing(
    estimated_frame: pd.DataFrame,
    terms: DeliverableTerms,
) -> None:
    first = _run(estimated_frame, terms)
    second = _run(estimated_frame, terms)

    assert first == second
    assert first.completed and first.downstream_eligible
    assert first.estimate_provenance_bound
    assert verify_basket_risk_hedge_hash_chain(first)
    assert b'"downstream_eligible": true' in basket_risk_hedge_report_json(first)
    assert first.maximum_absolute_accounting_gap is not None
    assert first.maximum_absolute_accounting_gap < 1e-9
    assert first.maximum_normalized_exposure_residual is not None
    assert first.maximum_normalized_exposure_residual < 1e-12
    assert len(first.instrument_frame()) == 12
    assert len(first.component_frame()) == 8
    for snapshot in first.snapshots[:-1]:
        assert snapshot.solver_condition_number is not None
        assert snapshot.solver_condition_number < 100.0
        assert max(abs(value) for value in snapshot.residual_component_deltas) < 1e-10
        assert abs(snapshot.residual_spot_scale_gamma) < 1e-8
        assert abs(snapshot.residual_parallel_vega) < 1e-8
        assert abs(snapshot.residual_correlation_strength_sensitivity) < 1e-8
    assert all(item.target_position == 0.0 for item in first.snapshots[-1].hedge_instruments)
    assert all(item.target_position == 0.0 for item in first.snapshots[-1].components)

    tampered_snapshot = replace(first.snapshots[0], cash=first.snapshots[0].cash + 1.0)
    tampered = replace(first, snapshots=(tampered_snapshot, *first.snapshots[1:]))
    assert not verify_basket_risk_hedge_hash_chain(tampered)


def test_advanced_hedge_fails_closed_on_solver_and_provenance_limits(
    estimated_frame: pd.DataFrame,
    terms: DeliverableTerms,
) -> None:
    blocked = run_basket_risk_hedge_replay(
        estimated_frame,
        terms,
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=BasketRiskHedgeSettings(paths=2_000, maximum_condition_number=1.0),
    )
    assert not blocked.downstream_eligible
    assert "condition number" in blocked.failure_reasons[0]

    without_provenance = estimated_frame.drop(
        columns=[
            "input_store_snapshot_sha256",
            "input_manifest_hashes_json",
            "input_estimate_sha256",
            "input_estimate_json",
        ]
    )
    with pytest.raises(BasketRiskHedgeSchemaError, match="provenance is required"):
        _run(without_provenance, terms)


def test_advanced_hedge_rejects_definition_drift(
    estimated_frame: pd.DataFrame,
    terms: DeliverableTerms,
) -> None:
    changed = estimated_frame.copy()
    payload = json.loads(str(changed.loc[1, "hedge_instruments_json"]))
    payload[0]["strike"] += 1.0
    changed.loc[1, "hedge_instruments_json"] = json.dumps(payload)
    result = _run(changed, terms)
    assert not result.downstream_eligible
    assert "definitions cannot change" in result.failure_reasons[0]


def test_advanced_settings_validate_limits() -> None:
    with pytest.raises(ValueError, match="maximum_condition_number"):
        BasketRiskHedgeSettings(maximum_condition_number=0.0)
    with pytest.raises(ValueError, match="transaction costs"):
        BasketRiskHedgeSettings(component_cost_bps=-1.0)
