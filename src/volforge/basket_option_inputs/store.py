"""Append-only storage and point-in-time resolution for basket-option quotes."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from volforge.deliverables import parse_deliverable_terms
from volforge.models.common import OptionType

BASKET_OPTION_QUOTE_SCHEMA_VERSION = 1
BASKET_OPTION_QUOTE_MODES = ("initial", "incremental", "backfill", "correction")
BASKET_OPTION_QUOTE_COLUMNS = (
    "market_timestamp",
    "received_timestamp",
    "source_record_id",
    "revision",
    "record_action",
    "basket_id",
    "instrument_id",
    "role",
    "option_type",
    "strike",
    "expiry_timestamp",
    "bid",
    "ask",
    "settlement_method",
    "contract_multiplier",
    "currency",
    "underlying_id",
    "deliverable_json",
)
BASKET_OPTION_QUOTE_PROVENANCE_COLUMNS = (
    "input_option_quote_snapshot_sha256",
    "input_option_quote_snapshot_json",
    "input_option_quote_manifest_hashes_json",
)
SYNTHETIC_TARGET_INSTRUMENT_ID = "TARGET-CALL-202604"
MAX_OPTION_QUOTE_ROWS = 100_000
MAX_OPTION_QUOTE_ARTIFACT_BYTES = 20 * 1024 * 1024
_RECORD_COLUMNS = ("source", *BASKET_OPTION_QUOTE_COLUMNS)


class BasketOptionQuoteSchemaError(ValueError):
    """Raised when canonical option-quote records are malformed."""


class BasketOptionQuoteIntegrityError(RuntimeError):
    """Raised when append-only or point-in-time quote invariants fail."""


@dataclass(frozen=True, slots=True)
class BasketOptionQuoteBatch:
    """One append-only option-quote ingestion manifest."""

    batch_sequence: int
    batch_id: str
    schema_version: int
    source: str
    mode: str
    ingested_timestamp: str
    source_file_name: str
    source_sha256: str
    source_size_bytes: int
    raw_archive_name: str
    row_count: int
    target_count: int
    hedge_count: int
    minimum_market_timestamp: str
    maximum_market_timestamp: str
    reason: str | None
    previous_manifest_hash: str | None
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class BasketOptionQuoteVerification:
    """Integrity result for a quote database and its raw archives."""

    valid: bool
    schema_version: int
    batch_count: int
    record_count: int
    final_manifest_hash: str | None
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BasketOptionQuoteSnapshot:
    """Target and hedge quotes independently known at one formation time."""

    basket_id: str
    source: str
    formation_timestamp: str
    dataset_known_at: str
    target_instrument_id: str
    target_option_type: str
    target_strike: float
    target_expiry_timestamp: str
    target_bid: float
    target_ask: float
    target_settlement_method: str
    target_contract_multiplier: float
    target_currency: str
    target_underlying_id: str | None
    target_deliverable_json: str
    target_deliverable_sha256: str
    hedge_instruments_json: str
    quote_count: int
    source_record_ids: tuple[str, ...]
    record_payload_sha256s: tuple[str, ...]
    batch_ids: tuple[str, ...]
    manifest_hashes: tuple[str, ...]
    snapshot_sha256: str

    def _payload(self, *, include_hash: bool) -> dict[str, Any]:
        payload = asdict(self)
        for field in (
            "source_record_ids",
            "record_payload_sha256s",
            "batch_ids",
            "manifest_hashes",
        ):
            payload[field] = list(payload[field])
        if not include_hash:
            payload.pop("snapshot_sha256")
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible snapshot audit."""
        return self._payload(include_hash=True)

    @property
    def canonical_json(self) -> str:
        """Return the canonical self-hashed snapshot JSON."""
        return _canonical_json(self.to_dict())


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: bytes | str | object) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = _canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise BasketOptionQuoteSchemaError(f"{name} must be timezone-aware") from exc
    if pd.isna(result) or result.tzinfo is None:
        raise BasketOptionQuoteSchemaError(f"{name} must be timezone-aware")
    return result.tz_convert("UTC")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BasketOptionQuoteSchemaError(f"{name} must be a non-empty string")
    return value.strip()


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise BasketOptionQuoteSchemaError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BasketOptionQuoteSchemaError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise BasketOptionQuoteSchemaError(f"{name} must be finite")
    return result


def _revision(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise BasketOptionQuoteSchemaError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise BasketOptionQuoteSchemaError(f"{name} must be a non-negative integer") from exc
    if result < 0 or float(value) != result:
        raise BasketOptionQuoteSchemaError(f"{name} must be a non-negative integer")
    return result


def _normalize_records(frame: pd.DataFrame, *, source: str) -> list[dict[str, Any]]:
    missing = sorted(set(BASKET_OPTION_QUOTE_COLUMNS) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(BASKET_OPTION_QUOTE_COLUMNS))
    if missing:
        raise BasketOptionQuoteSchemaError("missing option-quote columns: " + ", ".join(missing))
    if extra:
        raise BasketOptionQuoteSchemaError("unsupported option-quote columns: " + ", ".join(extra))
    if not 1 <= len(frame) <= MAX_OPTION_QUOTE_ROWS:
        raise BasketOptionQuoteSchemaError(
            f"option-quote batch requires 1 to {MAX_OPTION_QUOTE_ROWS} rows"
        )
    records: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        prefix = f"row {position}"
        market = _timestamp(row["market_timestamp"], f"{prefix} market_timestamp")
        received = _timestamp(row["received_timestamp"], f"{prefix} received_timestamp")
        expiry = _timestamp(row["expiry_timestamp"], f"{prefix} expiry_timestamp")
        if market > received:
            raise BasketOptionQuoteSchemaError(
                f"{prefix} requires market_timestamp <= received_timestamp"
            )
        action = _text(row["record_action"], f"{prefix} record_action").lower()
        if action not in {"upsert", "cancel"}:
            raise BasketOptionQuoteSchemaError(f"{prefix} record_action must be upsert or cancel")
        role = _text(row["role"], f"{prefix} role").lower()
        if role not in {"target", "hedge"}:
            raise BasketOptionQuoteSchemaError(f"{prefix} role must be target or hedge")
        option_type = OptionType.parse(str(row["option_type"])).value
        strike = _finite(row["strike"], f"{prefix} strike")
        bid = _finite(row["bid"], f"{prefix} bid")
        ask = _finite(row["ask"], f"{prefix} ask")
        multiplier = _finite(row["contract_multiplier"], f"{prefix} contract_multiplier")
        underlying = None if pd.isna(row["underlying_id"]) else str(row["underlying_id"]).strip()
        if underlying == "":
            underlying = None
        try:
            terms = parse_deliverable_terms(
                row["deliverable_json"],
                settlement_method=str(row["settlement_method"]),
                contract_multiplier=multiplier,
                currency=str(row["currency"]),
                underlying_id=underlying,
            )
        except ValueError as exc:
            raise BasketOptionQuoteSchemaError(f"{prefix} contract is invalid: {exc}") from exc
        if strike < 0.0 or expiry < market:
            raise BasketOptionQuoteSchemaError(f"{prefix} strike/expiry is invalid")
        if bid < 0.0 or ask < bid:
            raise BasketOptionQuoteSchemaError(f"{prefix} requires 0 <= bid <= ask")
        if action == "cancel":
            bid = ask = 0.0
        record: dict[str, Any] = {
            "source": source,
            "market_timestamp": market.isoformat(),
            "received_timestamp": received.isoformat(),
            "source_record_id": _text(row["source_record_id"], f"{prefix} source_record_id"),
            "revision": _revision(row["revision"], f"{prefix} revision"),
            "record_action": action,
            "basket_id": _text(row["basket_id"], f"{prefix} basket_id"),
            "instrument_id": _text(row["instrument_id"], f"{prefix} instrument_id"),
            "role": role,
            "option_type": option_type,
            "strike": strike,
            "expiry_timestamp": expiry.isoformat(),
            "bid": bid,
            "ask": ask,
            "settlement_method": terms.settlement_method,
            "contract_multiplier": terms.contract_multiplier,
            "currency": terms.currency,
            "underlying_id": terms.underlying_id,
            "deliverable_json": terms.canonical_json,
        }
        record["payload_json"] = _canonical_json(record)
        record["payload_sha256"] = _sha256(record["payload_json"])
        records.append(record)
    identities = [(row["source_record_id"], row["revision"]) for row in records]
    if len(identities) != len(set(identities)):
        raise BasketOptionQuoteSchemaError("batch has duplicate source_record_id/revision pairs")
    package_roles: dict[tuple[str, str], int] = {}
    for row in records:
        if row["record_action"] == "upsert" and row["role"] == "target":
            key = (row["basket_id"], row["market_timestamp"])
            package_roles[key] = package_roles.get(key, 0) + 1
    if any(count > 1 for count in package_roles.values()):
        raise BasketOptionQuoteSchemaError("a basket/time package may contain only one target")
    return records


def _manifest_payload(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "schema_version",
            "source",
            "mode",
            "ingested_timestamp",
            "source_file_name",
            "source_sha256",
            "source_size_bytes",
            "raw_archive_name",
            "row_count",
            "target_count",
            "hedge_count",
            "minimum_market_timestamp",
            "maximum_market_timestamp",
            "reason",
            "previous_manifest_hash",
        )
    }


def verify_option_quote_snapshot(value: object) -> bool:
    """Return whether a serialized quote snapshot has a valid self-hash."""
    expected_fields = set(BasketOptionQuoteSnapshot.__dataclass_fields__)
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or not isinstance(value.get("snapshot_sha256"), str)
        or not isinstance(value.get("quote_count"), int)
        or value.get("quote_count", 0) < 2
        or any(
            not isinstance(value.get(field), list)
            for field in (
                "source_record_ids",
                "record_payload_sha256s",
                "batch_ids",
                "manifest_hashes",
            )
        )
    ):
        return False
    payload = dict(value)
    digest = payload.pop("snapshot_sha256")
    try:
        int(digest, 16)
    except ValueError:
        return False
    return len(digest) == 64 and _sha256(payload) == digest


class BasketOptionQuoteStore:
    """SQLite-backed append-only store for independently sourced option quotes."""

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = Path(path).resolve()
        self.raw_directory = self.path.with_suffix(self.path.suffix + ".raw")
        if not self.path.is_file():
            raise FileNotFoundError(f"option-quote store does not exist: {self.path}")
        if not self.raw_directory.is_dir():
            raise FileNotFoundError(
                f"option-quote raw archive does not exist: {self.raw_directory}"
            )

    @classmethod
    def create(cls, path: str | PathLike[str]) -> BasketOptionQuoteStore:
        """Create a new store with database-level append-only triggers."""
        resolved = Path(path).resolve()
        raw = resolved.with_suffix(resolved.suffix + ".raw")
        if resolved.exists() or raw.exists():
            raise FileExistsError(f"refusing to overwrite option-quote store: {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(resolved)
        try:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE option_quote_batches (
                    batch_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL UNIQUE, schema_version INTEGER NOT NULL,
                    source TEXT NOT NULL, mode TEXT NOT NULL, ingested_timestamp TEXT NOT NULL,
                    source_file_name TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                    source_size_bytes INTEGER NOT NULL, raw_archive_name TEXT NOT NULL,
                    row_count INTEGER NOT NULL, target_count INTEGER NOT NULL,
                    hedge_count INTEGER NOT NULL, minimum_market_timestamp TEXT NOT NULL,
                    maximum_market_timestamp TEXT NOT NULL, reason TEXT,
                    previous_manifest_hash TEXT, manifest_hash TEXT NOT NULL UNIQUE,
                    UNIQUE(source, source_sha256)
                );
                CREATE TABLE option_quote_records (
                    record_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL REFERENCES option_quote_batches(batch_id),
                    ingested_timestamp TEXT NOT NULL, source TEXT NOT NULL,
                    market_timestamp TEXT NOT NULL, received_timestamp TEXT NOT NULL,
                    source_record_id TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision >= 0),
                    record_action TEXT NOT NULL CHECK(record_action IN ('upsert','cancel')),
                    basket_id TEXT NOT NULL, instrument_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('target','hedge')), option_type TEXT NOT NULL,
                    strike REAL NOT NULL, expiry_timestamp TEXT NOT NULL,
                    bid REAL NOT NULL, ask REAL NOT NULL, settlement_method TEXT NOT NULL,
                    contract_multiplier REAL NOT NULL, currency TEXT NOT NULL,
                    underlying_id TEXT, deliverable_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                    UNIQUE(source, source_record_id, revision)
                );
                CREATE INDEX option_quote_knowledge_idx ON option_quote_records(
                    basket_id, source, instrument_id, market_timestamp,
                    received_timestamp, ingested_timestamp
                );
                CREATE TRIGGER option_quote_batches_no_update BEFORE UPDATE ON option_quote_batches
                BEGIN SELECT RAISE(ABORT, 'option_quote_batches is append-only'); END;
                CREATE TRIGGER option_quote_batches_no_delete BEFORE DELETE ON option_quote_batches
                BEGIN SELECT RAISE(ABORT, 'option_quote_batches is append-only'); END;
                CREATE TRIGGER option_quote_records_no_update BEFORE UPDATE ON option_quote_records
                BEGIN SELECT RAISE(ABORT, 'option_quote_records is append-only'); END;
                CREATE TRIGGER option_quote_records_no_delete BEFORE DELETE ON option_quote_records
                BEGIN SELECT RAISE(ABORT, 'option_quote_records is append-only'); END;
                """
            )
            connection.commit()
        finally:
            connection.close()
        raw.mkdir(parents=True, exist_ok=False)
        return cls(resolved)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def ingest_csv(
        self,
        source_path: str | PathLike[str],
        *,
        source: str,
        mode: Literal["initial", "incremental", "backfill", "correction"],
        ingested_timestamp: str | pd.Timestamp,
        reason: str | None = None,
    ) -> BasketOptionQuoteBatch:
        """Validate, archive, hash, and atomically append a canonical CSV."""
        path = Path(source_path)
        raw = path.read_bytes()
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise BasketOptionQuoteSchemaError(f"cannot parse option-quote CSV: {exc}") from exc
        return self._ingest(
            frame,
            raw_bytes=raw,
            source_file_name=path.name,
            source=source,
            mode=mode,
            ingested_timestamp=ingested_timestamp,
            reason=reason,
        )

    def ingest_frame(
        self,
        frame: pd.DataFrame,
        *,
        source_file_name: str,
        source: str,
        mode: Literal["initial", "incremental", "backfill", "correction"],
        ingested_timestamp: str | pd.Timestamp,
        reason: str | None = None,
    ) -> BasketOptionQuoteBatch:
        """Append a frame with a deterministic CSV raw artifact."""
        raw = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode()
        return self._ingest(
            frame,
            raw_bytes=raw,
            source_file_name=Path(source_file_name).name,
            source=source,
            mode=mode,
            ingested_timestamp=ingested_timestamp,
            reason=reason,
        )

    def _ingest(
        self,
        frame: pd.DataFrame,
        *,
        raw_bytes: bytes,
        source_file_name: str,
        source: str,
        mode: str,
        ingested_timestamp: str | pd.Timestamp,
        reason: str | None,
    ) -> BasketOptionQuoteBatch:
        source_name = _text(source, "source")
        if mode not in BASKET_OPTION_QUOTE_MODES:
            raise ValueError(f"mode must be one of {', '.join(BASKET_OPTION_QUOTE_MODES)}")
        if not source_file_name or Path(source_file_name).name != source_file_name:
            raise ValueError("source_file_name must be a base name")
        if len(raw_bytes) > MAX_OPTION_QUOTE_ARTIFACT_BYTES:
            raise BasketOptionQuoteSchemaError("option-quote artifact is too large")
        reason_text = reason.strip() if reason is not None else None
        if mode in {"backfill", "correction"} and not reason_text:
            raise ValueError(f"{mode} ingestion requires a non-empty reason")
        ingested = _timestamp(ingested_timestamp, "ingested_timestamp")
        records = _normalize_records(frame, source=source_name)
        if any(
            _timestamp(row["received_timestamp"], "received_timestamp") > ingested
            for row in records
        ):
            raise BasketOptionQuoteIntegrityError(
                "received_timestamp cannot follow the batch ingested_timestamp"
            )
        market_times = [row["market_timestamp"] for row in records]
        source_digest = _sha256(raw_bytes)
        archive_name = f"{source_digest}.csv"
        archive_path = self.raw_directory / archive_name
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = connection.execute(
                "SELECT COUNT(*) count, MAX(market_timestamp) latest FROM option_quote_records WHERE source = ?",
                (source_name,),
            ).fetchone()
            source_exists = int(state["count"]) > 0
            if mode == "initial" and source_exists:
                raise BasketOptionQuoteIntegrityError(
                    "initial mode requires an empty source history"
                )
            if mode != "initial" and not source_exists:
                raise BasketOptionQuoteIntegrityError(
                    f"{mode} mode requires an existing source history"
                )
            if mode == "incremental" and state["latest"] and min(market_times) < state["latest"]:
                raise BasketOptionQuoteIntegrityError(
                    "incremental ingestion would silently backfill; use backfill mode"
                )
            if connection.execute(
                "SELECT 1 FROM option_quote_batches WHERE source = ? AND source_sha256 = ?",
                (source_name, source_digest),
            ).fetchone():
                raise BasketOptionQuoteIntegrityError(
                    "this exact source artifact was already ingested"
                )
            latest_ingestion = connection.execute(
                "SELECT MAX(ingested_timestamp) FROM option_quote_batches"
            ).fetchone()[0]
            if latest_ingestion and ingested.isoformat() < latest_ingestion:
                raise BasketOptionQuoteIntegrityError(
                    "ingested_timestamp cannot precede the latest ingestion"
                )
            identifiers = sorted({str(row["source_record_id"]) for row in records})
            placeholders = ",".join("?" for _ in identifiers)
            existing_rows = connection.execute(
                f"""
                WITH ranked AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY source_record_id ORDER BY revision DESC
                    ) rank FROM option_quote_records
                    WHERE source = ? AND source_record_id IN ({placeholders})
                ) SELECT * FROM ranked WHERE rank = 1
                """,
                (source_name, *identifiers),
            ).fetchall()
            existing = {str(row["source_record_id"]): row for row in existing_rows}
            if mode == "correction" and set(identifiers) != set(existing):
                raise BasketOptionQuoteIntegrityError(
                    "correction mode requires every source_record_id to exist"
                )
            if mode != "correction" and existing:
                raise BasketOptionQuoteIntegrityError(
                    "existing source_record_id requires correction mode"
                )
            identity_columns = (
                "market_timestamp",
                "basket_id",
                "instrument_id",
                "role",
                "option_type",
                "strike",
                "expiry_timestamp",
                "settlement_method",
                "contract_multiplier",
                "currency",
                "underlying_id",
                "deliverable_json",
            )
            grouped: dict[str, list[dict[str, Any]]] = {}
            for record in records:
                grouped.setdefault(str(record["source_record_id"]), []).append(record)
            for identifier, revisions in grouped.items():
                revisions.sort(key=lambda item: int(item["revision"]))
                prior = existing.get(identifier)
                previous_revision = int(prior["revision"]) if prior is not None else -1
                previous_received = str(prior["received_timestamp"]) if prior is not None else None
                active = prior is not None and prior["record_action"] == "upsert"
                identity_reference = prior if prior is not None else revisions[0]
                for record in revisions:
                    if int(record["revision"]) <= previous_revision:
                        raise BasketOptionQuoteIntegrityError(
                            "quote revisions must increase strictly"
                        )
                    if (
                        prior is None
                        and previous_revision == -1
                        and (record["revision"] != 0 or record["record_action"] != "upsert")
                    ):
                        raise BasketOptionQuoteIntegrityError(
                            "new quote identities require revision zero upserts"
                        )
                    if any(
                        record[column] != identity_reference[column] for column in identity_columns
                    ):
                        raise BasketOptionQuoteIntegrityError(
                            "quote identity and contract fields cannot change across revisions"
                        )
                    if (
                        previous_received is not None
                        and record["received_timestamp"] < previous_received
                    ):
                        raise BasketOptionQuoteIntegrityError(
                            "received_timestamp cannot move backward"
                        )
                    if record["record_action"] == "cancel" and not active:
                        raise BasketOptionQuoteIntegrityError("cannot cancel an inactive quote")
                    previous_revision = int(record["revision"])
                    previous_received = str(record["received_timestamp"])
                    active = record["record_action"] == "upsert"
            # An instrument identifier has one immutable contract definition per source/basket.
            contract_columns = identity_columns[2:]
            for record in records:
                prior_contracts = connection.execute(
                    """
                    SELECT * FROM option_quote_records
                    WHERE source = ? AND basket_id = ? AND instrument_id = ? LIMIT 1
                    """,
                    (source_name, record["basket_id"], record["instrument_id"]),
                ).fetchone()
                if prior_contracts is not None and any(
                    record[column] != prior_contracts[column] for column in contract_columns
                ):
                    raise BasketOptionQuoteIntegrityError(
                        f"contract definition for {record['instrument_id']!r} cannot drift"
                    )
            if archive_path.exists():
                if _sha256(archive_path.read_bytes()) != source_digest:
                    raise BasketOptionQuoteIntegrityError("raw archive content-address mismatch")
            else:
                archive_path.write_bytes(raw_bytes)
            previous_row = connection.execute(
                "SELECT manifest_hash FROM option_quote_batches ORDER BY batch_sequence DESC LIMIT 1"
            ).fetchone()
            previous = str(previous_row[0]) if previous_row else None
            fields: dict[str, Any] = {
                "schema_version": BASKET_OPTION_QUOTE_SCHEMA_VERSION,
                "source": source_name,
                "mode": mode,
                "ingested_timestamp": ingested.isoformat(),
                "source_file_name": source_file_name,
                "source_sha256": source_digest,
                "source_size_bytes": len(raw_bytes),
                "raw_archive_name": archive_name,
                "row_count": len(records),
                "target_count": sum(row["role"] == "target" for row in records),
                "hedge_count": sum(row["role"] == "hedge" for row in records),
                "minimum_market_timestamp": min(market_times),
                "maximum_market_timestamp": max(market_times),
                "reason": reason_text,
                "previous_manifest_hash": previous,
            }
            manifest_hash = _sha256(_manifest_payload(fields))
            batch_id = manifest_hash
            connection.execute(
                """
                INSERT INTO option_quote_batches(
                    batch_id, schema_version, source, mode, ingested_timestamp,
                    source_file_name, source_sha256, source_size_bytes, raw_archive_name,
                    row_count, target_count, hedge_count, minimum_market_timestamp,
                    maximum_market_timestamp, reason, previous_manifest_hash, manifest_hash
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    batch_id,
                    fields["schema_version"],
                    source_name,
                    mode,
                    ingested.isoformat(),
                    source_file_name,
                    source_digest,
                    len(raw_bytes),
                    archive_name,
                    len(records),
                    fields["target_count"],
                    fields["hedge_count"],
                    min(market_times),
                    max(market_times),
                    reason_text,
                    previous,
                    manifest_hash,
                ),
            )
            columns = (*_RECORD_COLUMNS, "payload_json", "payload_sha256")
            values = ",".join("?" for _ in range(len(columns) + 2))
            for record in records:
                connection.execute(
                    f"""INSERT INTO option_quote_records(
                        batch_id, ingested_timestamp, {", ".join(columns)}
                    ) VALUES ({values})""",
                    (batch_id, ingested.isoformat(), *(record[column] for column in columns)),
                )
            sequence = int(
                connection.execute(
                    "SELECT batch_sequence FROM option_quote_batches WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()[0]
            )
            connection.commit()
        return BasketOptionQuoteBatch(
            batch_sequence=sequence,
            batch_id=batch_id,
            **fields,
            manifest_hash=manifest_hash,
        )

    def batches(self) -> tuple[BasketOptionQuoteBatch, ...]:
        """Return ingestion manifests in append order."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM option_quote_batches ORDER BY batch_sequence"
            ).fetchall()
        return tuple(
            BasketOptionQuoteBatch(
                **{field: row[field] for field in BasketOptionQuoteBatch.__dataclass_fields__}
            )
            for row in rows
        )

    def records_as_known(
        self,
        formation_at: str | pd.Timestamp,
        *,
        basket_id: str | None = None,
        source: str | None = None,
        dataset_known_at: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return latest active quote revisions known by formation and dataset cutoffs."""
        formation = _timestamp(formation_at, "formation_at")
        cutoff = (
            _timestamp(dataset_known_at, "dataset_known_at").isoformat()
            if dataset_known_at is not None
            else "9999-12-31T23:59:59+00:00"
        )
        conditions = [
            "r.market_timestamp <= ?",
            "r.received_timestamp <= ?",
            "r.ingested_timestamp <= ?",
        ]
        parameters: list[Any] = [formation.isoformat(), formation.isoformat(), cutoff]
        if basket_id is not None:
            conditions.append("r.basket_id = ?")
            parameters.append(_text(basket_id, "basket_id"))
        if source is not None:
            conditions.append("r.source = ?")
            parameters.append(_text(source, "source"))
        query = f"""
            WITH eligible AS (
                SELECT r.*, b.batch_sequence, b.manifest_hash,
                    ROW_NUMBER() OVER (
                        PARTITION BY r.source, r.source_record_id
                        ORDER BY r.revision DESC, r.received_timestamp DESC,
                                 r.ingested_timestamp DESC
                    ) revision_rank
                FROM option_quote_records r JOIN option_quote_batches b ON b.batch_id = r.batch_id
                WHERE {" AND ".join(conditions)}
            ) SELECT * FROM eligible
            WHERE revision_rank = 1 AND record_action = 'upsert'
            ORDER BY market_timestamp, source, basket_id, instrument_id
        """
        with self._connection() as connection:
            frame = pd.read_sql_query(query, connection, params=parameters)
        for column in (
            "market_timestamp",
            "received_timestamp",
            "ingested_timestamp",
            "expiry_timestamp",
        ):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame.drop(columns=["revision_rank"], errors="ignore")

    def resolve_snapshot(
        self,
        basket_id: str,
        formation_timestamp: str | pd.Timestamp,
        *,
        target_instrument_id: str,
        hedge_instrument_ids: Sequence[str],
        source: str | None = None,
        maximum_age: str | pd.Timedelta = "15min",
        dataset_known_at: str | pd.Timestamp | None = None,
        _store_verified: bool = False,
    ) -> BasketOptionQuoteSnapshot:
        """Resolve the newest active quote for each explicitly requested instrument."""
        if not _store_verified:
            verification = self.verify()
            if not verification.valid:
                raise BasketOptionQuoteIntegrityError(
                    "option-quote store failed integrity verification: "
                    + "; ".join(verification.failure_reasons)
                )
        formation = _timestamp(formation_timestamp, "formation_timestamp")
        age_limit = pd.to_timedelta(maximum_age)
        if age_limit < pd.Timedelta(0):
            raise ValueError("maximum_age must be non-negative")
        target_id = _text(target_instrument_id, "target_instrument_id")
        hedge_ids = tuple(_text(value, "hedge_instrument_id") for value in hedge_instrument_ids)
        if not hedge_ids or len(hedge_ids) != len(set(hedge_ids)) or target_id in hedge_ids:
            raise ValueError("target and hedge instrument identifiers must be non-empty and unique")
        frame = self.records_as_known(
            formation,
            basket_id=basket_id,
            source=source,
            dataset_known_at=dataset_known_at,
        )
        if frame.empty:
            raise BasketOptionQuoteIntegrityError("no option quotes were known at formation")
        sources = sorted(frame["source"].unique().tolist())
        if source is None and len(sources) != 1:
            raise BasketOptionQuoteIntegrityError(
                "quote resolution requires source when multiple sources are eligible"
            )
        selected_source = _text(source or sources[0], "source")
        frame = frame.loc[frame["source"] == selected_source]
        requested = (target_id, *hedge_ids)
        selected_rows: list[pd.Series] = []
        for instrument_id in requested:
            history = frame.loc[frame["instrument_id"] == instrument_id]
            if history.empty:
                raise BasketOptionQuoteIntegrityError(
                    f"no active quote for instrument {instrument_id!r} at formation"
                )
            newest = history.sort_values(
                ["market_timestamp", "received_timestamp", "revision", "record_sequence"]
            ).iloc[-1]
            age = formation - newest["market_timestamp"]
            if age > age_limit:
                raise BasketOptionQuoteIntegrityError(
                    f"quote for {instrument_id!r} is stale by {age.total_seconds():.6g}s"
                )
            selected_rows.append(newest)
        target = selected_rows[0]
        if target["role"] != "target" or any(row["role"] != "hedge" for row in selected_rows[1:]):
            raise BasketOptionQuoteIntegrityError(
                "requested instrument roles do not match the store"
            )

        def instrument_payload(row: pd.Series) -> dict[str, Any]:
            return {
                "instrument_id": str(row["instrument_id"]),
                "option_type": str(row["option_type"]),
                "strike": float(row["strike"]),
                "expiry_timestamp": row["expiry_timestamp"].isoformat(),
                "bid": float(row["bid"]),
                "ask": float(row["ask"]),
                "settlement_method": str(row["settlement_method"]),
                "contract_multiplier": float(row["contract_multiplier"]),
                "currency": str(row["currency"]),
                "underlying_id": (
                    None if pd.isna(row["underlying_id"]) else str(row["underlying_id"])
                ),
                "deliverable_json": str(row["deliverable_json"]),
            }

        selected_frame = pd.DataFrame(selected_rows)
        batch_ids = tuple(sorted(selected_frame["batch_id"].astype(str).unique()))
        with self._connection() as connection:
            placeholders = ",".join("?" for _ in batch_ids)
            manifests = connection.execute(
                f"SELECT batch_id, manifest_hash FROM option_quote_batches WHERE batch_id IN ({placeholders})",
                batch_ids,
            ).fetchall()
        manifest_map = {str(row["batch_id"]): str(row["manifest_hash"]) for row in manifests}
        terms = parse_deliverable_terms(
            target["deliverable_json"],
            settlement_method=str(target["settlement_method"]),
            contract_multiplier=float(target["contract_multiplier"]),
            currency=str(target["currency"]),
            underlying_id=None
            if pd.isna(target["underlying_id"])
            else str(target["underlying_id"]),
        )
        snapshot = BasketOptionQuoteSnapshot(
            basket_id=_text(basket_id, "basket_id"),
            source=selected_source,
            formation_timestamp=formation.isoformat(),
            dataset_known_at=(
                _timestamp(dataset_known_at, "dataset_known_at").isoformat()
                if dataset_known_at is not None
                else "latest"
            ),
            target_instrument_id=target_id,
            target_option_type=str(target["option_type"]),
            target_strike=float(target["strike"]),
            target_expiry_timestamp=target["expiry_timestamp"].isoformat(),
            target_bid=float(target["bid"]),
            target_ask=float(target["ask"]),
            target_settlement_method=terms.settlement_method,
            target_contract_multiplier=terms.contract_multiplier,
            target_currency=terms.currency,
            target_underlying_id=terms.underlying_id,
            target_deliverable_json=terms.canonical_json,
            target_deliverable_sha256=terms.sha256,
            hedge_instruments_json=_canonical_json(
                [instrument_payload(row) for row in selected_rows[1:]]
            ),
            quote_count=len(selected_rows),
            source_record_ids=tuple(str(row["source_record_id"]) for row in selected_rows),
            record_payload_sha256s=tuple(str(row["payload_sha256"]) for row in selected_rows),
            batch_ids=batch_ids,
            manifest_hashes=tuple(manifest_map[value] for value in batch_ids),
            snapshot_sha256="",
        )
        return replace(
            snapshot,
            snapshot_sha256=_sha256(snapshot._payload(include_hash=False)),
        )

    def materialize_schedule(
        self,
        schedule: pd.DataFrame,
        *,
        basket_id: str,
        target_instrument_id: str,
        hedge_instrument_ids: Sequence[str],
        source: str | None = None,
        maximum_age: str | pd.Timedelta = "15min",
        dataset_known_at: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Resolve a formation/rate schedule into quote- and provenance-bound rows."""
        required = {"formation_timestamp", "risk_free_rate"}
        missing = sorted(required - set(schedule.columns))
        if missing:
            raise BasketOptionQuoteSchemaError(
                "missing option-quote schedule columns: " + ", ".join(missing)
            )
        verification = self.verify()
        if not verification.valid:
            raise BasketOptionQuoteIntegrityError(
                "option-quote store failed integrity verification: "
                + "; ".join(verification.failure_reasons)
            )
        rows: list[dict[str, Any]] = []
        for _, row in schedule.iterrows():
            snapshot = self.resolve_snapshot(
                basket_id,
                row["formation_timestamp"],
                target_instrument_id=target_instrument_id,
                hedge_instrument_ids=hedge_instrument_ids,
                source=source,
                maximum_age=maximum_age,
                dataset_known_at=dataset_known_at,
                _store_verified=True,
            )
            rows.append(
                {
                    "formation_timestamp": snapshot.formation_timestamp,
                    "option_bid": snapshot.target_bid,
                    "option_ask": snapshot.target_ask,
                    "risk_free_rate": _finite(row["risk_free_rate"], "risk_free_rate"),
                    "hedge_instruments_json": snapshot.hedge_instruments_json,
                    "input_option_quote_snapshot_sha256": snapshot.snapshot_sha256,
                    "input_option_quote_snapshot_json": snapshot.canonical_json,
                    "input_option_quote_manifest_hashes_json": _canonical_json(
                        snapshot.manifest_hashes
                    ),
                }
            )
        return pd.DataFrame(rows)

    def verify(self) -> BasketOptionQuoteVerification:
        """Verify database, payload hashes, raw archives, and manifest chain."""
        failures: list[str] = []
        try:
            with self._connection() as connection:
                if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    failures.append("SQLite quick_check failed")
                batches = connection.execute(
                    "SELECT * FROM option_quote_batches ORDER BY batch_sequence"
                ).fetchall()
                records = connection.execute(
                    "SELECT * FROM option_quote_records ORDER BY record_sequence"
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            return BasketOptionQuoteVerification(
                False, BASKET_OPTION_QUOTE_SCHEMA_VERSION, 0, 0, None, (str(exc),)
            )
        previous: str | None = None
        expected_archives: set[str] = set()
        for row in batches:
            fields = {key: row[key] for key in _manifest_payload(dict(row))}
            expected = _sha256(fields)
            if row["schema_version"] != BASKET_OPTION_QUOTE_SCHEMA_VERSION:
                failures.append(f"batch {row['batch_sequence']} has unsupported schema version")
            if row["previous_manifest_hash"] != previous or row["manifest_hash"] != expected:
                failures.append(f"batch {row['batch_sequence']} manifest chain/hash mismatch")
            previous = str(row["manifest_hash"])
            expected_archives.add(str(row["raw_archive_name"]))
            archive = self.raw_directory / str(row["raw_archive_name"])
            if not archive.is_file():
                failures.append(f"raw archive {archive.name} is missing")
            else:
                content = archive.read_bytes()
                if (
                    len(content) != row["source_size_bytes"]
                    or _sha256(content) != row["source_sha256"]
                ):
                    failures.append(f"raw archive {archive.name} size/hash mismatch")
        actual_archives = {path.name for path in self.raw_directory.iterdir() if path.is_file()}
        if actual_archives != expected_archives:
            failures.append("raw archive directory contains missing or unmanifested files")
        for row in records:
            payload = str(row["payload_json"])
            if _sha256(payload) != row["payload_sha256"]:
                failures.append(f"record {row['record_sequence']} payload hash mismatch")
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                failures.append(f"record {row['record_sequence']} payload JSON is invalid")
                continue
            expected = {column: row[column] for column in _RECORD_COLUMNS}
            if parsed != expected:
                failures.append(f"record {row['record_sequence']} payload/column mismatch")
        return BasketOptionQuoteVerification(
            valid=not failures,
            schema_version=BASKET_OPTION_QUOTE_SCHEMA_VERSION,
            batch_count=len(batches),
            record_count=len(records),
            final_manifest_hash=previous,
            failure_reasons=tuple(failures),
        )


def generate_synthetic_option_quote_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate six independently ingestible quote records per smile-replay formation."""
    from volforge.basket_estimation import generate_synthetic_smile_basket_inputs
    from volforge.basket_inputs import SYNTHETIC_BASKET_ID
    from volforge.basket_replay import SYNTHETIC_BASKET_DELIVERABLE, SYNTHETIC_BASKET_EXPIRY

    _, advanced_schedule = generate_synthetic_smile_basket_inputs()
    records: list[dict[str, Any]] = []
    hedge_ids: list[str] = []
    for sequence, (_, row) in enumerate(advanced_schedule.iterrows(), start=1):
        formation = _timestamp(row["formation_timestamp"], "formation_timestamp")
        market = formation - pd.Timedelta(minutes=2)
        received = formation - pd.Timedelta(minutes=1)
        quotes = [
            {
                "instrument_id": SYNTHETIC_TARGET_INSTRUMENT_ID,
                "role": "target",
                "option_type": "call",
                "strike": 60.0,
                "expiry_timestamp": SYNTHETIC_BASKET_EXPIRY,
                "bid": float(row["option_bid"]),
                "ask": float(row["option_ask"]),
                "settlement_method": "physical",
                "contract_multiplier": 100.0,
                "currency": "USD",
                "underlying_id": "ACME",
                "deliverable_json": SYNTHETIC_BASKET_DELIVERABLE,
            },
            *json.loads(str(row["hedge_instruments_json"])),
        ]
        if not hedge_ids:
            hedge_ids.extend(str(quote["instrument_id"]) for quote in quotes[1:])
        for quote in quotes:
            instrument_id = str(quote["instrument_id"])
            records.append(
                {
                    "market_timestamp": market.isoformat(),
                    "received_timestamp": received.isoformat(),
                    "source_record_id": f"SYN-OPTION-{sequence:04d}:{instrument_id}",
                    "revision": 0,
                    "record_action": "upsert",
                    "basket_id": SYNTHETIC_BASKET_ID,
                    **quote,
                    "role": quote.get("role", "hedge"),
                    "deliverable_json": (
                        quote["deliverable_json"]
                        if isinstance(quote["deliverable_json"], str)
                        else _canonical_json(quote["deliverable_json"])
                    ),
                }
            )
    schedule = advanced_schedule[["formation_timestamp", "risk_free_rate"]].copy()
    schedule.attrs["target_instrument_id"] = SYNTHETIC_TARGET_INSTRUMENT_ID
    schedule.attrs["hedge_instrument_ids"] = tuple(hedge_ids)
    return pd.DataFrame(records, columns=BASKET_OPTION_QUOTE_COLUMNS), schedule


def basket_option_quote_report_json(
    value: BasketOptionQuoteBatch | BasketOptionQuoteVerification | BasketOptionQuoteSnapshot,
) -> bytes:
    """Serialize a readable deterministic quote-store audit."""
    payload = value.to_dict() if isinstance(value, BasketOptionQuoteSnapshot) else asdict(value)
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


__all__ = [
    "BASKET_OPTION_QUOTE_COLUMNS",
    "BASKET_OPTION_QUOTE_MODES",
    "BASKET_OPTION_QUOTE_PROVENANCE_COLUMNS",
    "BASKET_OPTION_QUOTE_SCHEMA_VERSION",
    "SYNTHETIC_TARGET_INSTRUMENT_ID",
    "BasketOptionQuoteBatch",
    "BasketOptionQuoteIntegrityError",
    "BasketOptionQuoteSchemaError",
    "BasketOptionQuoteSnapshot",
    "BasketOptionQuoteStore",
    "BasketOptionQuoteVerification",
    "basket_option_quote_report_json",
    "generate_synthetic_option_quote_inputs",
    "verify_option_quote_snapshot",
]
