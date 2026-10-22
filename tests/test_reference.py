import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from volforge.history import HistoricalEventStore, HistoricalIntegrityError
from volforge.reference import ReferenceDataStore, ReferenceIntegrityError


def product_row(
    *,
    published: str = "2024-12-01T00:00:00Z",
) -> dict[str, object]:
    return {
        "source_record_id": "product-xyz",
        "revision": 0,
        "record_action": "upsert",
        "published_timestamp": published,
        "effective_from": "2024-01-01T00:00:00Z",
        "product_id": "XYZ-EQUITY-OPTIONS",
        "product_symbol": "XYZ",
        "exchange": "TEST",
        "calendar_id": "TEST-RTH",
        "underlying_asset_type": "equity",
        "exercise_style": "american",
        "settlement_method": "physical",
        "settlement_style": "pm",
        "currency": "USD",
        "default_multiplier": 100,
        "expiration_rule": "listed expiration timestamp",
        "last_trade_rule": "listed last-trade timestamp",
        "settlement_rule": "physical delivery after exercise",
        "status": "active",
        "source_reference": "test product circular",
    }


def contract_row(
    *,
    source_record_id: str = "contract-xyz-100c",
    symbol: str = "XYZ-20250321-C-100",
    published: str = "2024-12-01T00:00:00Z",
    effective_from: str = "2024-01-01T00:00:00Z",
    effective_to: str | None = None,
    deliverable_quantity: float = 100.0,
    corporate_action_id: str | None = None,
    expiration: str = "2025-03-21T20:00:00Z",
) -> dict[str, object]:
    return {
        "source_record_id": source_record_id,
        "revision": 0,
        "record_action": "upsert",
        "published_timestamp": published,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "contract_id": "XYZ-20250321-C-100-STABLE",
        "market_data_source": "demo",
        "source_contract_symbol": symbol,
        "product_id": "XYZ-EQUITY-OPTIONS",
        "underlying_id": "XYZ-US",
        "underlying_symbol": "XYZ",
        "option_type": "call",
        "strike": 100.0,
        "expiration_timestamp": expiration,
        "last_trade_timestamp": expiration,
        "settlement_timestamp": expiration,
        "contract_multiplier": 100.0,
        "deliverable_json": (
            '[{"asset_id":"XYZ-US","asset_type":"security",'
            f'"quantity":{deliverable_quantity},"currency":"USD"}}]'
        ),
        "status": "active",
        "corporate_action_id": corporate_action_id,
        "source_reference": "test series file",
    }


def session_row(
    session_date: str,
    open_timestamp: str | None,
    close_timestamp: str | None,
    *,
    source_record_id: str | None = None,
    status: str = "open",
    reason: str | None = None,
) -> dict[str, object]:
    return {
        "source_record_id": source_record_id or f"session-{session_date}",
        "revision": 0,
        "record_action": "upsert",
        "published_timestamp": "2024-12-01T00:00:00Z",
        "calendar_id": "TEST-RTH",
        "session_date": session_date,
        "session_name": "RTH",
        "status": status,
        "timezone": "America/New_York",
        "open_timestamp": open_timestamp,
        "close_timestamp": close_timestamp,
        "reason": reason,
        "source_reference": "test exchange calendar",
    }


def quote_row(
    record_id: str,
    event_time: str,
    received_time: str,
    *,
    symbol: str = "XYZ-20250321-C-100",
    expiration: str = "2025-03-21T20:00:00Z",
) -> dict[str, object]:
    return {
        "record_type": "quote",
        "event_timestamp": event_time,
        "received_timestamp": received_time,
        "source_record_id": record_id,
        "revision": 0,
        "record_action": "upsert",
        "underlying_symbol": "XYZ",
        "contract_symbol": symbol,
        "expiration": expiration,
        "strike": 100.0,
        "option_type": "call",
        "bid": 4.9,
        "ask": 5.1,
        "underlying_spot": 100.0,
        "risk_free_rate": 0.03,
        "dividend_yield": 0.01,
        "exchange": "TEST",
    }


def seed_reference_store(
    store: ReferenceDataStore,
    *,
    sessions: list[dict[str, object]] | None = None,
    contracts: list[dict[str, object]] | None = None,
) -> None:
    store.ingest_frame(
        pd.DataFrame([product_row()]),
        source_file_name="products.csv",
        kind="product",
        source="test-reference",
        mode="initial",
        ingested_timestamp="2025-01-02T14:00:00Z",
    )
    store.ingest_frame(
        pd.DataFrame(
            sessions
            or [
                session_row(
                    "2025-01-02",
                    "2025-01-02T14:30:00Z",
                    "2025-01-02T21:00:00Z",
                )
            ]
        ),
        source_file_name="sessions.csv",
        kind="session",
        source="test-reference",
        mode="initial",
        ingested_timestamp="2025-01-02T14:01:00Z",
    )
    store.ingest_frame(
        pd.DataFrame(contracts or [contract_row()]),
        source_file_name="contracts.csv",
        kind="contract",
        source="test-reference",
        mode="initial",
        ingested_timestamp="2025-01-02T14:02:00Z",
    )


def test_reference_gated_historical_replay_uses_stable_contracts(tmp_path: Path) -> None:
    reference = ReferenceDataStore.create(tmp_path / "reference.sqlite")
    seed_reference_store(reference)
    history = HistoricalEventStore.create(tmp_path / "history.sqlite")
    history.ingest_frame(
        pd.DataFrame(
            [
                quote_row(
                    "q1",
                    "2025-01-02T15:30:00Z",
                    "2025-01-02T15:30:00.100Z",
                ),
                quote_row(
                    "q2",
                    "2025-01-02T15:31:00Z",
                    "2025-01-02T15:31:00.100Z",
                ),
            ]
        ),
        source_file_name="quotes.csv",
        source="demo",
        mode="initial",
        ingested_timestamp="2025-01-02T16:00:00Z",
    )

    with pytest.raises(HistoricalIntegrityError, match="reference store"):
        history.replay_chain(
            ["2025-01-02T15:30:30Z", "2025-01-02T15:31:30Z"],
            source="demo",
            maximum_quote_age="2min",
        )
    replay = history.replay_chain(
        ["2025-01-02T15:30:30Z", "2025-01-02T15:31:30Z"],
        source="demo",
        maximum_quote_age="2min",
        reference_store=reference,
    )
    assert replay["quote_timestamp"].nunique() == 2
    assert replay["reference_validated"].all()
    assert replay["stable_contract_id"].unique().tolist() == ["XYZ-20250321-C-100-STABLE"]
    assert replay["contract_multiplier"].unique().tolist() == [100.0]
    assert replay["settlement_currency"].unique().tolist() == ["USD"]
    assert replay["formation_session_name"].unique().tolist() == ["RTH"]
    assert reference.verify().valid


def test_reference_store_preserves_cash_in_lieu_component(tmp_path: Path) -> None:
    reference = ReferenceDataStore.create(tmp_path / "reference.sqlite")
    adjusted = contract_row()
    adjusted["deliverable_json"] = (
        '[{"asset_id":"XYZ-US","asset_type":"security","quantity":50,'
        '"currency":"USD"},{"asset_id":"USD-CIL","asset_type":"cash_in_lieu",'
        '"quantity":12.5,"currency":"USD"}]'
    )
    seed_reference_store(reference, contracts=[adjusted])

    resolved = reference.records_as_known("contract", source="test-reference")

    assert '"asset_type":"cash_in_lieu"' in resolved.loc[0, "deliverable_json"]
    assert reference.verify().valid


def test_reference_resolution_rejects_closed_and_after_hours_quotes(tmp_path: Path) -> None:
    reference = ReferenceDataStore.create(tmp_path / "reference.sqlite")
    seed_reference_store(
        reference,
        sessions=[
            session_row(
                "2025-01-02",
                "2025-01-02T14:30:00Z",
                "2025-01-02T21:00:00Z",
            ),
            session_row(
                "2025-01-03",
                None,
                None,
                status="closed",
                reason="exchange holiday",
            ),
        ],
    )
    base = pd.DataFrame(
        [
            {
                **quote_row(
                    "q1",
                    "2025-01-03T15:30:00Z",
                    "2025-01-03T15:30:00.100Z",
                ),
                "source": "demo",
                "source_event_timestamp": pd.Timestamp("2025-01-03T15:30:00Z"),
                "quote_timestamp": pd.Timestamp("2025-01-03T15:30:30Z"),
            }
        ]
    )
    with pytest.raises(ReferenceIntegrityError, match="closed session"):
        reference.enrich_snapshot(base, "2025-01-03T15:30:30Z")

    after_hours = base.copy()
    after_hours["source_event_timestamp"] = pd.Timestamp("2025-01-02T22:00:00Z")
    after_hours["quote_timestamp"] = pd.Timestamp("2025-01-02T22:00:30Z")
    with pytest.raises(ReferenceIntegrityError, match="outside an open session"):
        reference.enrich_snapshot(after_hours, "2025-01-02T22:00:30Z")


def test_future_corporate_action_terms_do_not_leak_into_earlier_snapshots(
    tmp_path: Path,
) -> None:
    reference = ReferenceDataStore.create(tmp_path / "reference.sqlite")
    reference.ingest_frame(
        pd.DataFrame([product_row()]),
        source_file_name="products.csv",
        kind="product",
        source="test-reference",
        mode="initial",
        ingested_timestamp="2025-06-05T10:00:00Z",
    )
    reference.ingest_frame(
        pd.DataFrame(
            [
                session_row(
                    "2025-06-02",
                    "2025-06-02T13:30:00Z",
                    "2025-06-02T20:00:00Z",
                ),
                session_row(
                    "2025-06-04",
                    "2025-06-04T13:30:00Z",
                    "2025-06-04T20:00:00Z",
                ),
            ]
        ),
        source_file_name="sessions.csv",
        kind="session",
        source="test-reference",
        mode="initial",
        ingested_timestamp="2025-06-05T10:01:00Z",
    )
    action = {
        "source_record_id": "action-reverse-split",
        "revision": 0,
        "record_action": "upsert",
        "published_timestamp": "2025-06-03T12:00:00Z",
        "corporate_action_id": "CA-XYZ-2025-01",
        "underlying_id": "XYZ-US",
        "action_type": "reverse_split",
        "effective_timestamp": "2025-06-01T00:00:00Z",
        "old_symbol": "XYZ",
        "new_symbol": "XYZ",
        "description": "Illustrative two-for-one deliverable adjustment",
        "source_reference": "test OCC memo",
    }
    reference.ingest_frame(
        pd.DataFrame([action]),
        source_file_name="actions.csv",
        kind="corporate_action",
        source="test-reference",
        mode="initial",
        ingested_timestamp="2025-06-05T10:02:00Z",
    )
    adjusted_expiration = "2025-09-19T20:00:00Z"
    old_contract = contract_row(effective_to="2025-06-01T00:00:00Z", expiration=adjusted_expiration)
    new_contract = contract_row(
        source_record_id="contract-xyz-100c-adjusted",
        symbol="XYZ1-20250919-C-100",
        published="2025-06-03T12:00:00Z",
        effective_from="2025-06-01T00:00:00Z",
        deliverable_quantity=50.0,
        corporate_action_id="CA-XYZ-2025-01",
        expiration=adjusted_expiration,
    )
    reference.ingest_frame(
        pd.DataFrame([old_contract, new_contract]),
        source_file_name="contracts.csv",
        kind="contract",
        source="test-reference",
        mode="initial",
        ingested_timestamp="2025-06-05T10:03:00Z",
    )

    def adjusted_snapshot(timestamp: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    **quote_row(
                        "q-adjusted",
                        timestamp,
                        timestamp,
                        symbol="XYZ1-20250919-C-100",
                        expiration=adjusted_expiration,
                    ),
                    "source": "demo",
                    "source_event_timestamp": pd.Timestamp(timestamp),
                    "quote_timestamp": pd.Timestamp(timestamp),
                }
            ]
        )

    with pytest.raises(ReferenceIntegrityError, match="eligible contract"):
        reference.enrich_snapshot(
            adjusted_snapshot("2025-06-02T15:30:00Z"),
            "2025-06-02T15:30:00Z",
        )
    resolved = reference.enrich_snapshot(
        adjusted_snapshot("2025-06-04T15:30:00Z"),
        "2025-06-04T15:30:00Z",
    )
    assert resolved.loc[0, "stable_contract_id"] == "XYZ-20250321-C-100-STABLE"
    assert resolved.loc[0, "corporate_action_id"] == "CA-XYZ-2025-01"
    assert resolved.loc[0, "reference_action_payload_sha256"]
    assert '"quantity":50.0' in resolved.loc[0, "deliverable_json"]


def test_reference_store_is_append_only_and_detects_raw_tampering(tmp_path: Path) -> None:
    reference = ReferenceDataStore.create(tmp_path / "reference.sqlite")
    batch = reference.ingest_frame(
        pd.DataFrame([product_row()]),
        source_file_name="products.csv",
        kind="product",
        source="test-reference",
        mode="initial",
        ingested_timestamp="2025-01-02T14:00:00Z",
    )
    with (
        sqlite3.connect(reference.path) as connection,
        pytest.raises(sqlite3.IntegrityError, match="append-only"),
    ):
        connection.execute("UPDATE reference_records SET revision = 9")
    archive = reference.raw_directory / batch.raw_archive_name
    archive.write_bytes(archive.read_bytes() + b"tampered")
    verification = reference.verify()
    assert not verification.valid
    assert any("raw archive hash mismatch" in reason for reason in verification.failure_reasons)


def test_reference_revisions_and_backfills_remain_point_in_time(
    tmp_path: Path,
) -> None:
    reference = ReferenceDataStore.create(tmp_path / "reference.sqlite")
    original = product_row(published="2025-01-01T00:00:00Z")
    reference.ingest_frame(
        pd.DataFrame([original]),
        source_file_name="products.csv",
        kind="product",
        source="test-reference",
        mode="initial",
        ingested_timestamp="2025-01-02T00:00:00Z",
    )
    correction = {
        **original,
        "revision": 1,
        "published_timestamp": "2025-01-05T00:00:00Z",
        "settlement_rule": "corrected physical-delivery rule",
    }
    with pytest.raises(ValueError, match="reason"):
        reference.ingest_frame(
            pd.DataFrame([correction]),
            source_file_name="correction.csv",
            kind="product",
            source="test-reference",
            mode="correction",
            ingested_timestamp="2025-01-06T00:00:00Z",
        )
    reference.ingest_frame(
        pd.DataFrame([correction]),
        source_file_name="correction.csv",
        kind="product",
        source="test-reference",
        mode="correction",
        ingested_timestamp="2025-01-06T00:00:00Z",
        reason="exchange corrected the published settlement description",
    )
    before = reference.records_as_known("product", published_at="2025-01-04T00:00:00Z")
    after = reference.records_as_known("product", published_at="2025-01-07T00:00:00Z")
    assert before.loc[0, "settlement_rule"] == "physical delivery after exercise"
    assert after.loc[0, "settlement_rule"] == "corrected physical-delivery rule"

    older = {
        **product_row(published="2025-01-01T00:00:00Z"),
        "source_record_id": "product-older",
        "product_id": "OLDER-EQUITY-OPTIONS",
        "product_symbol": "OLD",
        "effective_from": "2023-01-01T00:00:00Z",
    }
    with pytest.raises(ReferenceIntegrityError, match="silently backfill"):
        reference.ingest_frame(
            pd.DataFrame([older]),
            source_file_name="older.csv",
            kind="product",
            source="test-reference",
            mode="incremental",
            ingested_timestamp="2025-01-07T00:00:00Z",
        )
    reference.ingest_frame(
        pd.DataFrame([older]),
        source_file_name="older.csv",
        kind="product",
        source="test-reference",
        mode="backfill",
        ingested_timestamp="2025-01-07T00:00:00Z",
        reason="late legacy product partition",
    )
    assert reference.verify().valid
