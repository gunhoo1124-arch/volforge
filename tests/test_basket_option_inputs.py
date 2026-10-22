import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from volforge.basket_covariance import materialize_selected_covariance_replay
from volforge.basket_estimation import generate_synthetic_smile_basket_inputs
from volforge.basket_inputs import SYNTHETIC_BASKET_ID, BasketInputStore
from volforge.basket_option_inputs import (
    SYNTHETIC_TARGET_INSTRUMENT_ID,
    BasketOptionQuoteIntegrityError,
    BasketOptionQuoteSchemaError,
    BasketOptionQuoteStore,
    generate_synthetic_option_quote_inputs,
    verify_option_quote_snapshot,
)
from volforge.basket_replay import SYNTHETIC_BASKET_DELIVERABLE, SYNTHETIC_BASKET_EXPIRY
from volforge.basket_risk_hedging import (
    BasketRiskHedgeSchemaError,
    BasketRiskHedgeSettings,
    run_basket_risk_hedge_replay,
    verify_basket_risk_hedge_hash_chain,
)
from volforge.deliverables import parse_deliverable_terms

OPTION_SOURCE = "synthetic-option-source"
COMPONENT_SOURCE = "synthetic-component-source"


def _seed_quote_store(
    path: Path,
) -> tuple[BasketOptionQuoteStore, pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    records, schedule = generate_synthetic_option_quote_inputs()
    hedge_ids = tuple(records.loc[records["role"] == "hedge", "instrument_id"].drop_duplicates())
    store = BasketOptionQuoteStore.create(path)
    store.ingest_frame(
        records,
        source_file_name="option-quotes.csv",
        source=OPTION_SOURCE,
        mode="initial",
        ingested_timestamp="2026-05-01T00:00:00Z",
    )
    return store, records, schedule, hedge_ids


def _terms():  # type: ignore[no-untyped-def]
    return parse_deliverable_terms(
        SYNTHETIC_BASKET_DELIVERABLE,
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ACME",
    )


def _integrated_replay_input(tmp_path: Path) -> pd.DataFrame:
    quote_store, _, schedule, hedge_ids = _seed_quote_store(tmp_path / "quotes.sqlite")
    quote_schedule = quote_store.materialize_schedule(
        schedule,
        basket_id=SYNTHETIC_BASKET_ID,
        source=OPTION_SOURCE,
        target_instrument_id=SYNTHETIC_TARGET_INSTRUMENT_ID,
        hedge_instrument_ids=hedge_ids,
    )
    component_records, _ = generate_synthetic_smile_basket_inputs()
    component_store = BasketInputStore.create(tmp_path / "components.sqlite")
    component_store.ingest_frame(
        component_records,
        source_file_name="components.csv",
        source=COMPONENT_SOURCE,
        mode="initial",
        ingested_timestamp="2026-05-01T00:00:00Z",
    )
    return materialize_selected_covariance_replay(
        component_store,
        quote_schedule,
        basket_id=SYNTHETIC_BASKET_ID,
        source=COMPONENT_SOURCE,
    )


def test_store_materializes_self_hashed_point_in_time_quotes(tmp_path: Path) -> None:
    store, records, schedule, hedge_ids = _seed_quote_store(tmp_path / "quotes.sqlite")

    verification = store.verify()
    materialized = store.materialize_schedule(
        schedule,
        basket_id=SYNTHETIC_BASKET_ID,
        source=OPTION_SOURCE,
        target_instrument_id=SYNTHETIC_TARGET_INSTRUMENT_ID,
        hedge_instrument_ids=hedge_ids,
    )
    snapshot = store.resolve_snapshot(
        SYNTHETIC_BASKET_ID,
        schedule.iloc[0]["formation_timestamp"],
        source=OPTION_SOURCE,
        target_instrument_id=SYNTHETIC_TARGET_INSTRUMENT_ID,
        hedge_instrument_ids=hedge_ids,
    )

    assert verification.valid
    assert verification.record_count == len(records) == 24
    assert len(store.batches()) == 1
    assert len(materialized) == len(schedule) == 4
    assert snapshot.quote_count == 6
    assert snapshot.target_instrument_id == SYNTHETIC_TARGET_INSTRUMENT_ID
    assert verify_option_quote_snapshot(snapshot.to_dict())
    assert json.loads(snapshot.hedge_instruments_json)[0]["instrument_id"] == hedge_ids[0]
    assert materialized["input_option_quote_snapshot_sha256"].str.len().eq(64).all()


def test_quote_corrections_obey_receipt_and_dataset_cutoffs(tmp_path: Path) -> None:
    store, records, _, _ = _seed_quote_store(tmp_path / "quotes.sqlite")
    identifier = "SYN-OPTION-0002:TARGET-CALL-202604"
    correction = records.loc[records["source_record_id"] == identifier].copy()
    correction.loc[:, "revision"] = 1
    correction.loc[:, "received_timestamp"] = "2026-04-10T20:30:00Z"
    correction.loc[:, "bid"] = correction["bid"] - 0.01
    store.ingest_frame(
        correction,
        source_file_name="correction.csv",
        source=OPTION_SOURCE,
        mode="correction",
        ingested_timestamp="2026-05-02T00:00:00Z",
        reason="provider corrected the executable bid",
    )

    before_receipt = store.records_as_known(
        "2026-04-10T20:00:00Z", basket_id=SYNTHETIC_BASKET_ID, source=OPTION_SOURCE
    )
    before_dataset = store.records_as_known(
        "2026-04-20T20:00:00Z",
        basket_id=SYNTHETIC_BASKET_ID,
        source=OPTION_SOURCE,
        dataset_known_at="2026-05-01T12:00:00Z",
    )
    after = store.records_as_known(
        "2026-04-20T20:00:00Z", basket_id=SYNTHETIC_BASKET_ID, source=OPTION_SOURCE
    )

    assert (
        before_receipt.loc[before_receipt["source_record_id"] == identifier, "revision"].item() == 0
    )
    assert (
        before_dataset.loc[before_dataset["source_record_id"] == identifier, "revision"].item() == 0
    )
    assert after.loc[after["source_record_id"] == identifier, "revision"].item() == 1


def test_cancellation_removes_a_required_instrument(tmp_path: Path) -> None:
    store, records, schedule, hedge_ids = _seed_quote_store(tmp_path / "quotes.sqlite")
    final_hedge = records.loc[
        (records["source_record_id"] == f"SYN-OPTION-0004:{hedge_ids[0]}")
    ].copy()
    final_hedge.loc[:, "revision"] = 1
    final_hedge.loc[:, "record_action"] = "cancel"
    store.ingest_frame(
        final_hedge,
        source_file_name="cancel.csv",
        source=OPTION_SOURCE,
        mode="correction",
        ingested_timestamp="2026-05-02T00:00:00Z",
        reason="market withdrawn",
    )

    with pytest.raises(BasketOptionQuoteIntegrityError, match="stale"):
        store.resolve_snapshot(
            SYNTHETIC_BASKET_ID,
            schedule.iloc[-1]["formation_timestamp"],
            source=OPTION_SOURCE,
            target_instrument_id=SYNTHETIC_TARGET_INSTRUMENT_ID,
            hedge_instrument_ids=hedge_ids,
        )


def test_stale_missing_and_ambiguous_quote_resolution_are_blocked(tmp_path: Path) -> None:
    store, _, schedule, hedge_ids = _seed_quote_store(tmp_path / "quotes.sqlite")
    with pytest.raises(BasketOptionQuoteIntegrityError, match="stale"):
        store.resolve_snapshot(
            SYNTHETIC_BASKET_ID,
            schedule.iloc[0]["formation_timestamp"],
            source=OPTION_SOURCE,
            target_instrument_id=SYNTHETIC_TARGET_INSTRUMENT_ID,
            hedge_instrument_ids=hedge_ids,
            maximum_age="30s",
        )
    with pytest.raises(BasketOptionQuoteIntegrityError, match="no active quote"):
        store.resolve_snapshot(
            SYNTHETIC_BASKET_ID,
            schedule.iloc[0]["formation_timestamp"],
            source=OPTION_SOURCE,
            target_instrument_id="MISSING",
            hedge_instrument_ids=hedge_ids,
        )
    second_records, _ = generate_synthetic_option_quote_inputs()
    store.ingest_frame(
        second_records,
        source_file_name="second-source.csv",
        source="second-source",
        mode="initial",
        ingested_timestamp="2026-05-02T00:00:00Z",
    )
    with pytest.raises(BasketOptionQuoteIntegrityError, match="requires source"):
        store.resolve_snapshot(
            SYNTHETIC_BASKET_ID,
            schedule.iloc[0]["formation_timestamp"],
            target_instrument_id=SYNTHETIC_TARGET_INSTRUMENT_ID,
            hedge_instrument_ids=hedge_ids,
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("role", "unknown", "role must"),
        ("bid", -1.0, "0 <= bid"),
        ("ask", -1.0, "0 <= bid"),
        ("record_action", "replace", "upsert or cancel"),
        ("option_type", "straddle", "option_type"),
        ("expiry_timestamp", "2020-01-01T00:00:00Z", "strike/expiry"),
        ("deliverable_json", "{}", "contract is invalid"),
    ],
)
def test_schema_rejects_malformed_quote_records(
    tmp_path: Path, column: str, value: object, message: str
) -> None:
    records, _ = generate_synthetic_option_quote_inputs()
    malformed = records.iloc[[0]].copy()
    malformed.loc[:, column] = value
    store = BasketOptionQuoteStore.create(tmp_path / "quotes.sqlite")
    with pytest.raises((BasketOptionQuoteSchemaError, ValueError), match=message):
        store.ingest_frame(
            malformed,
            source_file_name="bad.csv",
            source=OPTION_SOURCE,
            mode="initial",
            ingested_timestamp="2026-05-01T00:00:00Z",
        )


def test_ingestion_modes_and_contract_identity_are_enforced(tmp_path: Path) -> None:
    store, records, _, _ = _seed_quote_store(tmp_path / "quotes.sqlite")
    with pytest.raises(BasketOptionQuoteIntegrityError, match="empty source history"):
        store.ingest_frame(
            records,
            source_file_name="again.csv",
            source=OPTION_SOURCE,
            mode="initial",
            ingested_timestamp="2026-05-02T00:00:00Z",
        )
    new_record = records.iloc[[0]].copy()
    new_record.loc[:, "source_record_id"] = "late-new-record"
    new_record.loc[:, "market_timestamp"] = "2026-01-01T00:00:00Z"
    new_record.loc[:, "received_timestamp"] = "2026-01-01T00:01:00Z"
    with pytest.raises(BasketOptionQuoteIntegrityError, match="backfill"):
        store.ingest_frame(
            new_record,
            source_file_name="old.csv",
            source=OPTION_SOURCE,
            mode="incremental",
            ingested_timestamp="2026-05-02T00:00:00Z",
        )
    correction = records.iloc[[0]].copy()
    correction.loc[:, "revision"] = 1
    correction.loc[:, "strike"] = correction["strike"] + 1
    with pytest.raises(BasketOptionQuoteIntegrityError, match="cannot change"):
        store.ingest_frame(
            correction,
            source_file_name="drift.csv",
            source=OPTION_SOURCE,
            mode="correction",
            ingested_timestamp="2026-05-02T00:00:00Z",
            reason="bad contract drift",
        )
    new_record.loc[:, "market_timestamp"] = "2026-03-01T00:00:00Z"
    new_record.loc[:, "received_timestamp"] = "2026-03-01T00:01:00Z"
    with pytest.raises(ValueError, match="reason"):
        store.ingest_frame(
            new_record,
            source_file_name="backfill.csv",
            source=OPTION_SOURCE,
            mode="backfill",
            ingested_timestamp="2026-05-02T00:00:00Z",
        )


def test_store_is_append_only_and_raw_tampering_blocks_resolution(tmp_path: Path) -> None:
    store, _, schedule, hedge_ids = _seed_quote_store(tmp_path / "quotes.sqlite")
    with (
        sqlite3.connect(store.path) as connection,
        pytest.raises(sqlite3.IntegrityError, match="append-only"),
    ):
        connection.execute("UPDATE option_quote_records SET revision = 9")
    archive = store.raw_directory / store.batches()[0].raw_archive_name
    archive.write_bytes(archive.read_bytes() + b"tampered")

    verification = store.verify()
    assert not verification.valid
    assert any("raw archive" in reason for reason in verification.failure_reasons)
    with pytest.raises(BasketOptionQuoteIntegrityError, match="integrity verification"):
        store.resolve_snapshot(
            SYNTHETIC_BASKET_ID,
            schedule.iloc[0]["formation_timestamp"],
            source=OPTION_SOURCE,
            target_instrument_id=SYNTHETIC_TARGET_INSTRUMENT_ID,
            hedge_instrument_ids=hedge_ids,
        )


def test_quote_provenance_is_bound_into_advanced_replay(tmp_path: Path) -> None:
    replay_input = _integrated_replay_input(tmp_path)
    result = run_basket_risk_hedge_replay(
        replay_input,
        _terms(),
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=BasketRiskHedgeSettings(
            paths=2_000,
            include_smile_risks=True,
            require_covariance_selection=True,
            require_option_quote_provenance=True,
        ),
    )

    assert result.completed
    assert result.option_quote_provenance_bound
    assert all(snapshot.input_option_quote_snapshot_sha256 for snapshot in result.snapshots)
    assert verify_basket_risk_hedge_hash_chain(result)


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("option_bid", None, "target contract and quote"),
        ("input_option_quote_snapshot_sha256", "0" * 64, "does not match"),
        ("input_option_quote_manifest_hashes_json", "[]", "manifest identities"),
        ("hedge_instruments_json", "[]", "exactly 5"),
    ],
)
def test_advanced_replay_rejects_quote_lineage_tampering(
    tmp_path: Path, column: str, replacement: object, message: str
) -> None:
    replay_input = _integrated_replay_input(tmp_path)
    replay_input.loc[0, column] = (
        float(replay_input.loc[0, "option_bid"]) + 0.001 if column == "option_bid" else replacement
    )
    result = run_basket_risk_hedge_replay(
        replay_input,
        _terms(),
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=BasketRiskHedgeSettings(
            paths=2_000,
            include_smile_risks=True,
            require_option_quote_provenance=True,
        ),
    )

    assert not result.completed
    assert any(message in reason for reason in result.failure_reasons)


def test_advanced_replay_requires_complete_quote_provenance(tmp_path: Path) -> None:
    replay_input = _integrated_replay_input(tmp_path)
    without = replay_input.drop(columns=list(replay_input.filter(like="input_option_quote")))
    with pytest.raises(BasketRiskHedgeSchemaError, match="option-quote provenance is required"):
        run_basket_risk_hedge_replay(
            without,
            _terms(),
            option_type="call",
            strike=60,
            expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
            settings=BasketRiskHedgeSettings(require_option_quote_provenance=True),
        )
    partial = replay_input.drop(columns=["input_option_quote_snapshot_json"])
    with pytest.raises(BasketRiskHedgeSchemaError, match="incomplete option-quote provenance"):
        run_basket_risk_hedge_replay(
            partial,
            _terms(),
            option_type="call",
            strike=60,
            expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        )
