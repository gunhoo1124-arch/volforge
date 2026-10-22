import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from volforge.basket_inputs import (
    SYNTHETIC_BASKET_ID,
    BasketAdapterError,
    BasketInputIntegrityError,
    BasketInputSchemaError,
    BasketInputStore,
    HttpsJsonBasketInputAdapter,
    generate_synthetic_basket_inputs,
)
from volforge.basket_replay import (
    SYNTHETIC_BASKET_DELIVERABLE,
    SYNTHETIC_BASKET_EXPIRY,
    BasketReplaySettings,
    run_basket_hedge_replay,
    verify_basket_replay_hash_chain,
)
from volforge.deliverables import parse_deliverable_terms


def _seed_store(path: Path) -> tuple[BasketInputStore, pd.DataFrame, pd.DataFrame]:
    records, schedule = generate_synthetic_basket_inputs()
    store = BasketInputStore.create(path)
    store.ingest_frame(
        records,
        source_file_name="basket-inputs.csv",
        source="volforge-synthetic-v1",
        mode="initial",
        ingested_timestamp="2026-05-01T00:00:00Z",
    )
    return store, records, schedule


def test_store_materializes_provenance_bound_replay(tmp_path: Path) -> None:
    store, records, schedule = _seed_store(tmp_path / "basket.sqlite")

    verification = store.verify()
    replay_frame = store.materialize_replay(
        schedule,
        basket_id=SYNTHETIC_BASKET_ID,
        source="volforge-synthetic-v1",
    )
    terms = parse_deliverable_terms(
        SYNTHETIC_BASKET_DELIVERABLE,
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ACME",
    )
    result = run_basket_hedge_replay(
        replay_frame,
        terms,
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=BasketReplaySettings(
            paths=4_000,
            require_store_provenance=True,
        ),
    )

    assert verification.valid
    assert verification.record_count == len(records) == 12
    assert len(replay_frame) == 4
    assert replay_frame["input_store_snapshot_sha256"].str.len().eq(64).all()
    assert result.completed
    assert result.store_provenance_bound
    assert verify_basket_replay_hash_chain(result)


def test_received_cutoff_prevents_future_snapshot_selection(tmp_path: Path) -> None:
    records, _ = generate_synthetic_basket_inputs()
    final_id = "SYN-BASKET-0004"
    records.loc[
        records["snapshot_id"] == final_id,
        "received_timestamp",
    ] = "2026-04-30T20:01:00Z"
    store = BasketInputStore.create(tmp_path / "basket.sqlite")
    store.ingest_frame(
        records,
        source_file_name="basket-inputs.csv",
        source="volforge-synthetic-v1",
        mode="initial",
        ingested_timestamp="2026-05-01T00:00:00Z",
    )

    snapshot = store.resolve_snapshot(
        SYNTHETIC_BASKET_ID,
        "2026-04-30T20:00:00Z",
        source="volforge-synthetic-v1",
        maximum_age="30D",
    )

    assert snapshot.snapshot_id == "SYN-BASKET-0003"


def test_revisions_respect_received_and_dataset_cutoffs(tmp_path: Path) -> None:
    store, records, _ = _seed_store(tmp_path / "basket.sqlite")
    correction = records.loc[records["source_record_id"] == "SYN-BASKET-0002:component:ACME"].copy()
    correction.loc[:, "revision"] = 1
    correction.loc[:, "received_timestamp"] = "2026-04-10T20:30:00Z"
    correction.loc[:, "volatility"] = 0.5
    store.ingest_frame(
        correction,
        source_file_name="correction.csv",
        source="volforge-synthetic-v1",
        mode="correction",
        ingested_timestamp="2026-05-02T00:00:00Z",
        reason="provider corrected the component volatility",
    )

    before_receipt = store.records_as_known(
        "2026-04-10T20:00:00Z",
        basket_id=SYNTHETIC_BASKET_ID,
        source="volforge-synthetic-v1",
    )
    before_dataset_revision = store.records_as_known(
        "2026-04-20T20:00:00Z",
        basket_id=SYNTHETIC_BASKET_ID,
        source="volforge-synthetic-v1",
        dataset_known_at="2026-05-01T12:00:00Z",
    )
    after = store.records_as_known(
        "2026-04-20T20:00:00Z",
        basket_id=SYNTHETIC_BASKET_ID,
        source="volforge-synthetic-v1",
    )

    record_id = "SYN-BASKET-0002:component:ACME"
    assert (
        before_receipt.loc[before_receipt["source_record_id"] == record_id, "revision"].item() == 0
    )
    assert (
        before_dataset_revision.loc[
            before_dataset_revision["source_record_id"] == record_id, "revision"
        ].item()
        == 0
    )
    assert after.loc[after["source_record_id"] == record_id, "revision"].item() == 1
    assert after.loc[after["source_record_id"] == record_id, "volatility"].item() == 0.5


def test_store_rejects_incomplete_packages_and_replay_without_required_provenance(
    tmp_path: Path,
) -> None:
    records, schedule = generate_synthetic_basket_inputs()
    incomplete = records.drop(
        records.loc[records["record_type"] == "correlation"].index[0]
    ).reset_index(drop=True)
    store = BasketInputStore.create(tmp_path / "basket.sqlite")
    with pytest.raises(BasketInputSchemaError, match="incomplete correlation package"):
        store.ingest_frame(
            incomplete,
            source_file_name="incomplete.csv",
            source="test",
            mode="initial",
            ingested_timestamp="2026-05-01T00:00:00Z",
        )

    from volforge.basket_replay import generate_synthetic_basket_replay

    terms = parse_deliverable_terms(
        SYNTHETIC_BASKET_DELIVERABLE,
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ACME",
    )
    with pytest.raises(BasketInputIntegrityError, match="no basket-input records"):
        store.materialize_replay(schedule, basket_id=SYNTHETIC_BASKET_ID)
    with pytest.raises(ValueError, match="requires append-only store provenance"):
        run_basket_hedge_replay(
            generate_synthetic_basket_replay(),
            terms,
            option_type="call",
            strike=60,
            expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
            settings=BasketReplaySettings(require_store_provenance=True),
        )


def test_store_is_append_only_and_detects_raw_tampering(tmp_path: Path) -> None:
    store, _, _ = _seed_store(tmp_path / "basket.sqlite")
    with (
        sqlite3.connect(store.path) as connection,
        pytest.raises(sqlite3.IntegrityError, match="append-only"),
    ):
        connection.execute("UPDATE basket_input_records SET revision = 9")
    batch = store.batches()[0]
    archive = store.raw_directory / batch.raw_archive_name
    archive.write_bytes(archive.read_bytes() + b"tampered")

    verification = store.verify()

    assert not verification.valid
    assert any("raw archive" in reason for reason in verification.failure_reasons)
    with pytest.raises(BasketInputIntegrityError, match="integrity verification"):
        store.resolve_snapshot(
            SYNTHETIC_BASKET_ID,
            "2026-04-10T20:00:00Z",
            source="volforge-synthetic-v1",
        )


def test_snapshot_resolution_rejects_non_psd_correlation(tmp_path: Path) -> None:
    timestamp = "2026-04-01T19:58:00Z"
    received = "2026-04-01T19:59:00Z"
    rows: list[dict[str, object]] = []
    for asset_id, spot in (("A", 100.0), ("B", 80.0), ("C", 60.0)):
        rows.append(
            {
                "record_type": "component",
                "snapshot_id": "bad-psd",
                "market_timestamp": timestamp,
                "received_timestamp": received,
                "source_record_id": f"bad-psd:{asset_id}",
                "revision": 0,
                "record_action": "upsert",
                "basket_id": "BAD-PSD",
                "asset_id": asset_id,
                "paired_asset_id": None,
                "spot": spot,
                "volatility": 0.2,
                "dividend_yield": 0.0,
                "bid": spot - 0.1,
                "ask": spot + 0.1,
                "correlation": None,
            }
        )
    for left, right, correlation in (("A", "B", 0.9), ("A", "C", 0.9), ("B", "C", -0.9)):
        rows.append(
            {
                "record_type": "correlation",
                "snapshot_id": "bad-psd",
                "market_timestamp": timestamp,
                "received_timestamp": received,
                "source_record_id": f"bad-psd:{left}:{right}",
                "revision": 0,
                "record_action": "upsert",
                "basket_id": "BAD-PSD",
                "asset_id": left,
                "paired_asset_id": right,
                "spot": None,
                "volatility": None,
                "dividend_yield": None,
                "bid": None,
                "ask": None,
                "correlation": correlation,
            }
        )
    store = BasketInputStore.create(tmp_path / "bad-psd.sqlite")
    store.ingest_frame(
        pd.DataFrame(rows),
        source_file_name="bad-psd.csv",
        source="test",
        mode="initial",
        ingested_timestamp="2026-04-01T20:00:00Z",
    )

    with pytest.raises(BasketInputIntegrityError, match="positive semidefinite"):
        store.resolve_snapshot(
            "BAD-PSD",
            "2026-04-01T20:00:00Z",
            source="test",
        )


def test_https_adapter_is_bounded_and_does_not_retain_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, _ = generate_synthetic_basket_inputs()
    raw = json.dumps({"dataset_id": "demo-1", "records": records.to_dict("records")}).encode()
    captured: dict[str, object] = {}

    class Response:
        status = 200

        def __init__(self) -> None:
            self.headers = {
                "Content-Length": str(len(raw)),
                "Content-Type": "application/json",
                "ETag": '"abc"',
            }

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *args):  # type: ignore[no-untyped-def]
            return False

        def read(self, limit: int) -> bytes:
            assert limit > len(raw)
            return raw

    def fake_urlopen(request, **kwargs):  # type: ignore[no-untyped-def]
        captured["authorization"] = request.get_header("Authorization")
        captured.update(kwargs)
        return Response()

    monkeypatch.setenv("VOLFORGE_TEST_TOKEN", "secret-token")
    monkeypatch.setattr("volforge.basket_inputs.adapters.urllib.request.urlopen", fake_urlopen)
    adapter = HttpsJsonBasketInputAdapter(
        "https://data.example.test/v1/basket-inputs",
        authorization_env="VOLFORGE_TEST_TOKEN",
    )

    payload = adapter.fetch()

    assert len(payload.frame) == len(records)
    assert captured["authorization"] == "Bearer secret-token"
    assert payload.metadata["endpoint"] == "https://data.example.test/v1/basket-inputs"
    assert "secret-token" not in json.dumps(payload.metadata)
    with pytest.raises(ValueError, match="https"):
        replace(adapter, endpoint="http://data.example.test/inputs")
    monkeypatch.delenv("VOLFORGE_TEST_TOKEN")
    with pytest.raises(BasketAdapterError, match="not set"):
        adapter.fetch()
