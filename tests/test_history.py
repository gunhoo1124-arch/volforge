import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from volforge.history import (
    HistoricalEventStore,
    HistoricalIntegrityError,
    HistoricalSchemaError,
    history_report_json,
)


def quote_row(
    record_id: str,
    event_time: str,
    received_time: str,
    *,
    revision: int = 0,
    action: str = "upsert",
    strike: float = 100.0,
    bid: float = 4.9,
    ask: float = 5.1,
) -> dict[str, object]:
    return {
        "record_type": "quote",
        "event_timestamp": event_time,
        "received_timestamp": received_time,
        "source_record_id": record_id,
        "revision": revision,
        "record_action": action,
        "underlying_symbol": "XYZ",
        "contract_symbol": f"XYZ-20250321-C-{strike:g}",
        "expiration": "2025-03-21T20:00:00Z",
        "strike": strike,
        "option_type": "call",
        "bid": bid,
        "ask": ask,
        "bid_size": 10,
        "ask_size": 12,
        "underlying_spot": 100.0,
        "risk_free_rate": 0.03,
        "dividend_yield": 0.01,
        "exchange": "TEST",
    }


def trade_row(record_id: str, event_time: str, received_time: str) -> dict[str, object]:
    return {
        "record_type": "trade",
        "event_timestamp": event_time,
        "received_timestamp": received_time,
        "source_record_id": record_id,
        "revision": 0,
        "record_action": "upsert",
        "underlying_symbol": "XYZ",
        "contract_symbol": "XYZ-20250321-C-100",
        "expiration": "2025-03-21T20:00:00Z",
        "strike": 100.0,
        "option_type": "call",
        "trade_price": 5.0,
        "trade_size": 3,
        "trade_condition": "regular",
        "exchange": "TEST",
    }


def test_append_only_store_queries_quotes_trades_and_snapshots(tmp_path: Path) -> None:
    store = HistoricalEventStore.create(tmp_path / "history.sqlite")
    frame = pd.DataFrame(
        [
            quote_row(
                "q1",
                "2025-01-02T15:30:00Z",
                "2025-01-02T15:30:00.100Z",
                bid=4.8,
                ask=5.0,
            ),
            quote_row(
                "q2",
                "2025-01-02T15:31:00Z",
                "2025-01-02T15:31:01Z",
                bid=5.0,
                ask=5.2,
            ),
            trade_row(
                "t1",
                "2025-01-02T15:30:30Z",
                "2025-01-02T15:30:30.050Z",
            ),
        ]
    )
    batch = store.ingest_frame(
        frame,
        source_file_name="initial.csv",
        source="demo",
        mode="initial",
        ingested_timestamp="2025-01-02T16:00:00Z",
    )
    assert batch.row_count == 3
    assert batch.quote_count == 2
    assert batch.trade_count == 1
    assert batch.previous_manifest_hash is None

    early = store.chain_snapshot("2025-01-02T15:30:45Z", source="demo", maximum_quote_age="2min")
    assert early.iloc[0]["source_record_id"] == "q1"
    assert early.iloc[0]["quote_timestamp"] == pd.Timestamp("2025-01-02T15:30:45Z")
    assert early.iloc[0]["quote_age_seconds"] == pytest.approx(45.0)

    before_second_arrived = store.chain_snapshot(
        "2025-01-02T15:31:00Z", source="demo", maximum_quote_age="2min"
    )
    assert before_second_arrived.iloc[0]["source_record_id"] == "q1"
    later = store.chain_snapshot("2025-01-02T15:31:30Z", source="demo", maximum_quote_age="2min")
    assert later.iloc[0]["source_record_id"] == "q2"
    with pytest.raises(HistoricalIntegrityError, match="reference store"):
        store.replay_chain(
            ["2025-01-02T15:30:45Z", "2025-01-02T15:31:30Z"],
            source="demo",
            maximum_quote_age="2min",
        )

    trades = store.events_as_known(
        "trade",
        event_end="2025-01-02T15:31:00Z",
        received_at="2025-01-02T15:31:00Z",
        source="demo",
    )
    assert trades["source_record_id"].tolist() == ["t1"]
    verification = store.verify()
    assert verification.valid
    assert verification.batch_count == 1
    assert verification.event_count == 3
    assert b'"valid": true' in history_report_json(verification)


def test_received_and_ingested_cutoffs_preserve_revision_history(tmp_path: Path) -> None:
    store = HistoricalEventStore.create(tmp_path / "history.sqlite")
    original = pd.DataFrame(
        [
            quote_row(
                "q1",
                "2025-01-02T15:30:00Z",
                "2025-01-02T15:30:00.100Z",
                bid=4.8,
                ask=5.0,
            )
        ]
    )
    store.ingest_frame(
        original,
        source_file_name="original.csv",
        source="demo",
        mode="initial",
        ingested_timestamp="2025-01-02T16:00:00Z",
    )
    correction = pd.DataFrame(
        [
            quote_row(
                "q1",
                "2025-01-02T15:30:00Z",
                "2025-01-03T10:00:00Z",
                revision=1,
                bid=4.7,
                ask=4.9,
            )
        ]
    )
    corrected_batch = store.ingest_frame(
        correction,
        source_file_name="correction.csv",
        source="demo",
        mode="correction",
        ingested_timestamp="2025-01-03T11:00:00Z",
        reason="vendor corrected the ask",
    )
    assert corrected_batch.previous_manifest_hash is not None

    market_view_before_correction = store.events_as_known(
        "quote",
        received_at="2025-01-02T23:59:59Z",
        ingested_at="2025-01-04T00:00:00Z",
        source="demo",
    )
    assert market_view_before_correction.iloc[0]["ask"] == pytest.approx(5.0)
    dataset_view_before_ingestion = store.events_as_known(
        "quote",
        received_at="2025-01-04T00:00:00Z",
        ingested_at="2025-01-02T23:59:59Z",
        source="demo",
    )
    assert dataset_view_before_ingestion.iloc[0]["ask"] == pytest.approx(5.0)
    corrected = store.events_as_known(
        "quote",
        received_at="2025-01-04T00:00:00Z",
        ingested_at="2025-01-04T00:00:00Z",
        source="demo",
    )
    assert corrected.iloc[0]["revision"] == 1
    assert corrected.iloc[0]["ask"] == pytest.approx(4.9)

    cancellation = pd.DataFrame(
        [
            quote_row(
                "q1",
                "2025-01-02T15:30:00Z",
                "2025-01-04T10:00:00Z",
                revision=2,
                action="cancel",
            )
        ]
    )
    store.ingest_frame(
        cancellation,
        source_file_name="cancel.csv",
        source="demo",
        mode="correction",
        ingested_timestamp="2025-01-04T11:00:00Z",
        reason="vendor cancelled the quote",
    )
    assert store.events_as_known("quote", received_at="2025-01-05T00:00:00Z", source="demo").empty
    assert (
        store.events_as_known("quote", received_at="2025-01-03T12:00:00Z", source="demo").iloc[0][
            "revision"
        ]
        == 1
    )
    assert store.verify().valid


def test_backfill_is_explicit_and_tables_are_append_only(tmp_path: Path) -> None:
    store = HistoricalEventStore.create(tmp_path / "history.sqlite")
    initial = pd.DataFrame(
        [
            quote_row(
                "q2",
                "2025-01-02T15:31:00Z",
                "2025-01-02T15:31:00.100Z",
            )
        ]
    )
    store.ingest_frame(
        initial,
        source_file_name="initial.csv",
        source="demo",
        mode="initial",
        ingested_timestamp="2025-01-02T16:00:00Z",
    )
    older = pd.DataFrame(
        [
            quote_row(
                "q1",
                "2025-01-02T15:30:00Z",
                "2025-01-02T15:30:00.100Z",
            )
        ]
    )
    with pytest.raises(HistoricalIntegrityError, match="silently backfill"):
        store.ingest_frame(
            older,
            source_file_name="older.csv",
            source="demo",
            mode="incremental",
            ingested_timestamp="2025-01-02T16:01:00Z",
        )
    with pytest.raises(ValueError, match="reason"):
        store.ingest_frame(
            older,
            source_file_name="older.csv",
            source="demo",
            mode="backfill",
            ingested_timestamp="2025-01-02T16:01:00Z",
        )
    backfill = store.ingest_frame(
        older,
        source_file_name="older.csv",
        source="demo",
        mode="backfill",
        ingested_timestamp="2025-01-02T16:01:00Z",
        reason="late vendor partition",
    )
    assert backfill.mode == "backfill"
    assert len(store.batches()) == 2

    with sqlite3.connect(store.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE market_events SET bid = 0")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM ingestion_batches")


def test_tampered_raw_archive_and_invalid_timestamps_are_detected(tmp_path: Path) -> None:
    store = HistoricalEventStore.create(tmp_path / "history.sqlite")
    invalid = pd.DataFrame(
        [
            quote_row(
                "q1",
                "2025-01-02T15:30:00Z",
                "2025-01-02T15:29:59Z",
            )
        ]
    )
    with pytest.raises(HistoricalSchemaError, match="cannot precede"):
        store.ingest_frame(
            invalid,
            source_file_name="invalid.csv",
            source="demo",
            mode="initial",
            ingested_timestamp="2025-01-02T16:00:00Z",
        )

    valid = pd.DataFrame(
        [
            quote_row(
                "q1",
                "2025-01-02T15:30:00Z",
                "2025-01-02T15:30:00.100Z",
            )
        ]
    )
    batch = store.ingest_frame(
        valid,
        source_file_name="valid.csv",
        source="demo",
        mode="initial",
        ingested_timestamp="2025-01-02T16:00:00Z",
    )
    archive = store.raw_directory / batch.raw_archive_name
    archive.write_bytes(archive.read_bytes() + b"tampered")
    verification = store.verify()
    assert not verification.valid
    assert any("raw archive hash mismatch" in reason for reason in verification.failure_reasons)


def test_corrections_preserve_record_type_and_require_an_active_record(
    tmp_path: Path,
) -> None:
    store = HistoricalEventStore.create(tmp_path / "history.sqlite")
    original = pd.DataFrame(
        [
            quote_row(
                "q1",
                "2025-01-02T15:30:00Z",
                "2025-01-02T15:30:00.100Z",
            )
        ]
    )
    store.ingest_frame(
        original,
        source_file_name="original.csv",
        source="demo",
        mode="initial",
        ingested_timestamp="2025-01-02T16:00:00Z",
    )

    changed_type = trade_row(
        "q1",
        "2025-01-02T15:30:00Z",
        "2025-01-03T10:00:00Z",
    )
    changed_type["revision"] = 1
    with pytest.raises(HistoricalIntegrityError, match="record_type"):
        store.ingest_frame(
            pd.DataFrame([changed_type]),
            source_file_name="changed-type.csv",
            source="demo",
            mode="correction",
            ingested_timestamp="2025-01-03T11:00:00Z",
            reason="invalid vendor type change",
        )

    cancellation = pd.DataFrame(
        [
            quote_row(
                "q1",
                "2025-01-02T15:30:00Z",
                "2025-01-03T10:00:00Z",
                revision=1,
                action="cancel",
            )
        ]
    )
    store.ingest_frame(
        cancellation,
        source_file_name="cancel.csv",
        source="demo",
        mode="correction",
        ingested_timestamp="2025-01-03T11:00:00Z",
        reason="vendor cancellation",
    )
    repeated_cancellation = cancellation.copy()
    repeated_cancellation["revision"] = 2
    repeated_cancellation["received_timestamp"] = "2025-01-04T10:00:00Z"
    with pytest.raises(HistoricalIntegrityError, match="active prior upsert"):
        store.ingest_frame(
            repeated_cancellation,
            source_file_name="cancel-again.csv",
            source="demo",
            mode="correction",
            ingested_timestamp="2025-01-04T11:00:00Z",
            reason="duplicate vendor cancellation",
        )
