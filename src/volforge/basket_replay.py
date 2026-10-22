"""Point-in-time delta-hedge replay for explicit adjusted deliverables."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from typing import Any

import pandas as pd

from volforge.basket import (
    BasketMarket,
    BasketMonteCarloSettings,
    parse_basket_market,
    price_basket_option,
)
from volforge.deliverables import DeliverableTerms, parse_deliverable_terms, value_option_settlement
from volforge.models.common import OptionType

_SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0
MAX_BASKET_REPLAY_ROWS = 10_000
BASKET_REPLAY_COLUMNS = (
    "formation_timestamp",
    "market_timestamp",
    "received_timestamp",
    "source",
    "source_record_id",
    "option_bid",
    "option_ask",
    "risk_free_rate",
    "market_json",
    "execution_json",
)
BASKET_REPLAY_PROVENANCE_COLUMNS = (
    "input_store_snapshot_sha256",
    "input_manifest_hashes_json",
)
BASKET_REPLAY_ESTIMATION_COLUMNS = (
    "input_estimate_sha256",
    "input_estimate_json",
)

SYNTHETIC_BASKET_DELIVERABLE = (
    '[{"asset_id":"ACME","asset_type":"security","currency":"USD","quantity":50},'
    '{"asset_id":"SPIN","asset_type":"security","currency":"USD","quantity":10},'
    '{"asset_id":"USD-CIL","asset_type":"cash_in_lieu","currency":"USD","quantity":20}]'
)
SYNTHETIC_BASKET_EXPIRY = "2026-04-30T20:00:00+00:00"


class BasketReplaySchemaError(ValueError):
    """Raised when a basket replay table cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class BasketReplaySettings:
    """Sampling, execution, timing, and reconciliation controls."""

    paths: int = 20_000
    seed: int = 31
    antithetic: bool = True
    control_variate: bool = True
    confidence_level: float = 0.95
    short_option_contracts: float = 1.0
    component_cost_bps: float = 1.0
    option_cost_bps: float = 5.0
    maximum_market_age_seconds: float = 900.0
    accounting_tolerance: float = 1e-8
    require_store_provenance: bool = False
    require_estimated_inputs: bool = False

    def __post_init__(self) -> None:
        BasketMonteCarloSettings(
            paths=self.paths,
            seed=self.seed,
            antithetic=self.antithetic,
            control_variate=self.control_variate,
            confidence_level=self.confidence_level,
        )
        if not math.isfinite(self.short_option_contracts) or self.short_option_contracts <= 0.0:
            raise ValueError("short_option_contracts must be positive and finite")
        costs = (self.component_cost_bps, self.option_cost_bps)
        if any(not math.isfinite(value) or not 0.0 <= value <= 10_000.0 for value in costs):
            raise ValueError("transaction costs must lie between zero and 10,000 bps")
        if (
            not math.isfinite(self.maximum_market_age_seconds)
            or self.maximum_market_age_seconds < 0.0
        ):
            raise ValueError("maximum_market_age_seconds must be non-negative and finite")
        if not math.isfinite(self.accounting_tolerance) or self.accounting_tolerance < 0.0:
            raise ValueError("accounting_tolerance must be non-negative and finite")
        if not isinstance(self.require_store_provenance, bool):
            raise ValueError("require_store_provenance must be boolean")
        if not isinstance(self.require_estimated_inputs, bool):
            raise ValueError("require_estimated_inputs must be boolean")

    def monte_carlo(self, sequence: int) -> BasketMonteCarloSettings:
        """Return the deterministic sampling controls for one snapshot."""
        return BasketMonteCarloSettings(
            paths=self.paths,
            seed=self.seed + sequence,
            antithetic=self.antithetic,
            control_variate=self.control_variate,
            confidence_level=self.confidence_level,
        )


@dataclass(frozen=True, slots=True)
class ComponentExecutionQuote:
    """Executable component quote at one formation snapshot."""

    asset_id: str
    bid: float
    ask: float


@dataclass(frozen=True, slots=True)
class BasketComponentAudit:
    """One component's hedge target, trade, and execution accounting."""

    asset_id: str
    spot: float
    bid: float
    ask: float
    model_delta: float
    delta_standard_error: float
    prior_position: float
    target_position: float
    trade: float
    execution_price: float
    trade_cashflow: float
    spread_cost: float
    transaction_cost: float


@dataclass(frozen=True, slots=True)
class BasketReplaySnapshotAudit:
    """Tamper-evident pricing and self-financing record for one snapshot."""

    sequence: int
    formation_timestamp: str
    market_timestamp: str
    received_timestamp: str
    action: str
    source: str
    source_record_id: str
    snapshot_sha256: str
    market_sha256: str
    execution_sha256: str
    input_store_snapshot_sha256: str | None
    input_manifest_hashes: tuple[str, ...]
    input_estimate_sha256: str | None
    configuration_sha256: str
    previous_audit_hash: str | None
    audit_hash: str
    no_lookahead_passed: bool
    market_age_seconds: float
    maturity_years: float
    rate: float
    option_bid: float
    option_ask: float
    option_mid: float
    option_model_price: float
    option_model_standard_error: float
    option_model_error_to_mid: float
    pricing_bounds_passed: bool
    settlement_payoff: float | None
    option_cashflow: float
    cash_before: float
    cash_carry: float
    dividend_cashflow: float
    component_trade_cashflow: float
    spread_cost: float
    cumulative_spread_cost: float
    transaction_cost: float
    cumulative_transaction_cost: float
    turnover: float
    cumulative_turnover: float
    cash: float
    portfolio_value: float
    incremental_pnl: float
    cumulative_pnl: float
    accounting_gap: float
    components: tuple[BasketComponentAudit, ...]


@dataclass(frozen=True, slots=True)
class BasketReplayResult:
    """Completed or blocked point-in-time adjusted-option hedge replay."""

    settings: BasketReplaySettings
    option_type: str
    strike: float
    expiry_timestamp: str
    settlement_method: str
    classification: str
    contract_multiplier: float
    currency: str
    underlying_id: str | None
    deliverable_sha256: str
    configuration_sha256: str
    input_snapshot_count: int
    snapshots: tuple[BasketReplaySnapshotAudit, ...]
    completed: bool
    downstream_eligible: bool
    store_provenance_bound: bool
    estimated_input_provenance_bound: bool
    total_pnl: float | None
    total_spread_cost: float
    total_transaction_cost: float
    total_turnover: float
    maximum_absolute_accounting_gap: float | None
    final_audit_hash: str | None
    failure_reasons: tuple[str, ...]

    def ledger_frame(self) -> pd.DataFrame:
        """Return one flat row per processed replay snapshot."""
        rows: list[dict[str, Any]] = []
        for snapshot in self.snapshots:
            row = asdict(snapshot)
            components = row.pop("components")
            row["components_json"] = _canonical_json(components)
            rows.append(row)
        return pd.DataFrame(rows)

    def component_frame(self) -> pd.DataFrame:
        """Return one row per snapshot and hedge component."""
        rows: list[dict[str, Any]] = []
        for snapshot in self.snapshots:
            for component in snapshot.components:
                rows.append(
                    {
                        "sequence": snapshot.sequence,
                        "formation_timestamp": snapshot.formation_timestamp,
                        "action": snapshot.action,
                        **asdict(component),
                    }
                )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-compatible replay audit."""
        return asdict(self)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _aware_timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise BasketReplaySchemaError(f"{name} must be a valid timezone-aware timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise BasketReplaySchemaError(f"{name} must be a valid timezone-aware timestamp")
    return timestamp.tz_convert("UTC")


def _parse_execution_quotes(
    value: object,
    market: BasketMarket,
) -> tuple[tuple[ComponentExecutionQuote, ...], str, str]:
    if not isinstance(value, str):
        raise ValueError("execution_json must be a JSON string")
    if len(value.encode("utf-8")) > 1_048_576:
        raise ValueError("execution_json exceeds 1048576 bytes")
    try:
        parsed = json.loads(value, object_pairs_hook=_strict_object)
    except json.JSONDecodeError as exc:
        raise ValueError("execution_json is not valid JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError("execution_json must be a JSON list")
    quotes: list[ComponentExecutionQuote] = []
    for position, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"execution quote {position} must be an object")
        extra = sorted(set(item) - {"asset_id", "bid", "ask"})
        if extra:
            raise ValueError(
                f"execution quote {position} has unsupported fields: {', '.join(extra)}"
            )
        asset_id = item.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise ValueError(f"execution quote {position} asset_id must be a non-empty string")
        bid = _finite(item.get("bid"), f"bid for {asset_id}")
        ask = _finite(item.get("ask"), f"ask for {asset_id}")
        if bid <= 0.0 or ask < bid:
            raise ValueError(f"execution quote for {asset_id} requires 0 < bid <= ask")
        quotes.append(ComponentExecutionQuote(asset_id.strip(), bid, ask))
    quotes.sort(key=lambda quote: quote.asset_id)
    ids = tuple(quote.asset_id for quote in quotes)
    if len(ids) != len(set(ids)):
        raise ValueError("execution_json contains duplicate asset_id values")
    if ids != market.asset_ids:
        missing = sorted(set(market.asset_ids) - set(ids))
        unused = sorted(set(ids) - set(market.asset_ids))
        reasons: list[str] = []
        if missing:
            reasons.append("missing execution quotes: " + ", ".join(missing))
        if unused:
            reasons.append("unused execution quotes: " + ", ".join(unused))
        if not reasons:
            reasons.append("execution quotes must follow canonical asset order")
        raise ValueError("; ".join(reasons))
    for asset, quote in zip(market.assets, quotes, strict=True):
        if not quote.bid <= asset.spot <= quote.ask:
            raise ValueError(
                f"market spot for {asset.asset_id} must lie inside its execution quote"
            )
    canonical = _canonical_json([asdict(quote) for quote in quotes])
    return tuple(quotes), canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _option_quote(row: pd.Series) -> tuple[float, float, float]:
    bid = _finite(row["option_bid"], "option_bid")
    ask = _finite(row["option_ask"], "option_ask")
    if bid < 0.0 or ask < bid:
        raise ValueError("option quote requires 0 <= bid <= ask")
    return bid, ask, 0.5 * (bid + ask)


def _hex_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _store_provenance(
    row: pd.Series,
    *,
    supplied: bool,
) -> tuple[str | None, tuple[str, ...]]:
    if not supplied:
        return None, ()
    snapshot_hash = _hex_digest(row["input_store_snapshot_sha256"], "input_store_snapshot_sha256")
    raw_manifests = row["input_manifest_hashes_json"]
    if not isinstance(raw_manifests, str) or len(raw_manifests.encode("utf-8")) > 1_048_576:
        raise ValueError("input_manifest_hashes_json must be a bounded JSON string")
    try:
        parsed = json.loads(raw_manifests, object_pairs_hook=_strict_object)
    except json.JSONDecodeError as exc:
        raise ValueError("input_manifest_hashes_json is not valid JSON") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("input_manifest_hashes_json must be a non-empty JSON list")
    manifests = tuple(
        _hex_digest(value, f"input manifest hash {position}")
        for position, value in enumerate(parsed)
    )
    if len(manifests) != len(set(manifests)):
        raise ValueError("input_manifest_hashes_json contains duplicate hashes")
    return snapshot_hash, manifests


def _estimation_provenance(
    row: pd.Series,
    *,
    supplied: bool,
    market: BasketMarket,
    formation: pd.Timestamp,
    input_store_snapshot_sha256: str | None,
    input_manifest_hashes: tuple[str, ...],
) -> str | None:
    if not supplied:
        return None
    from volforge.basket_estimation import verify_basket_input_estimate

    estimate_hash = _hex_digest(row["input_estimate_sha256"], "input_estimate_sha256")
    raw_estimate = row["input_estimate_json"]
    if not isinstance(raw_estimate, str) or len(raw_estimate.encode("utf-8")) > 1_048_576:
        raise ValueError("input_estimate_json must be a bounded JSON string")
    try:
        payload = json.loads(raw_estimate, object_pairs_hook=_strict_object)
    except json.JSONDecodeError as exc:
        raise ValueError("input_estimate_json is not valid JSON") from exc
    if not isinstance(payload, dict) or not verify_basket_input_estimate(payload):
        raise ValueError("input estimate failed its self-hash verification")
    if payload.get("estimate_sha256") != estimate_hash:
        raise ValueError("input estimate hash does not match its provenance column")
    if payload.get("formation_timestamp") != formation.isoformat():
        raise ValueError("input estimate formation timestamp does not match replay")
    if payload.get("market_json") != market.canonical_json:
        raise ValueError("input estimate market does not match replay")
    if payload.get("input_store_snapshot_sha256") != input_store_snapshot_sha256:
        raise ValueError("input estimate store snapshot identity does not match replay")
    if payload.get("input_manifest_hashes") != list(input_manifest_hashes):
        raise ValueError("input estimate manifests do not match replay")
    return estimate_hash


def _snapshot_input_hash(
    *,
    formation: pd.Timestamp,
    market_timestamp: pd.Timestamp,
    received: pd.Timestamp,
    source: str,
    source_record_id: str,
    option_bid: float,
    option_ask: float,
    rate: float,
    market: BasketMarket,
    execution_json: str,
    input_store_snapshot_sha256: str | None,
    input_manifest_hashes: tuple[str, ...],
    input_estimate_sha256: str | None,
) -> str:
    return _sha256(
        {
            "formation_timestamp": formation.isoformat(),
            "market_timestamp": market_timestamp.isoformat(),
            "received_timestamp": received.isoformat(),
            "source": source,
            "source_record_id": source_record_id,
            "option_bid": option_bid,
            "option_ask": option_ask,
            "risk_free_rate": rate,
            "canonical_market_json": market.canonical_json,
            "canonical_execution_json": execution_json,
            "input_store_snapshot_sha256": input_store_snapshot_sha256,
            "input_manifest_hashes": input_manifest_hashes,
            "input_estimate_sha256": input_estimate_sha256,
        }
    )


def _audit_hash(snapshot: BasketReplaySnapshotAudit) -> str:
    payload = asdict(snapshot)
    payload.pop("audit_hash")
    return _sha256(payload)


def _configuration_hash(
    *,
    settings: BasketReplaySettings,
    option_type: str,
    strike: float,
    expiry_timestamp: str,
    settlement_method: str,
    classification: str,
    contract_multiplier: float,
    currency: str,
    underlying_id: str | None,
    deliverable_sha256: str,
    input_snapshot_count: int,
) -> str:
    return _sha256(
        {
            "settings": asdict(settings),
            "option_type": option_type,
            "strike": strike,
            "expiry_timestamp": expiry_timestamp,
            "settlement_method": settlement_method,
            "classification": classification,
            "contract_multiplier": contract_multiplier,
            "currency": currency,
            "underlying_id": underlying_id,
            "deliverable_sha256": deliverable_sha256,
            "input_snapshot_count": input_snapshot_count,
        }
    )


def verify_basket_replay_hash_chain(result: BasketReplayResult) -> bool:
    """Verify ordering, predecessor links, and every snapshot audit hash."""
    expected_configuration = _configuration_hash(
        settings=result.settings,
        option_type=result.option_type,
        strike=result.strike,
        expiry_timestamp=result.expiry_timestamp,
        settlement_method=result.settlement_method,
        classification=result.classification,
        contract_multiplier=result.contract_multiplier,
        currency=result.currency,
        underlying_id=result.underlying_id,
        deliverable_sha256=result.deliverable_sha256,
        input_snapshot_count=result.input_snapshot_count,
    )
    if result.configuration_sha256 != expected_configuration:
        return False
    previous: str | None = None
    for sequence, snapshot in enumerate(result.snapshots):
        if snapshot.sequence != sequence or snapshot.previous_audit_hash != previous:
            return False
        if snapshot.configuration_sha256 != expected_configuration:
            return False
        if _audit_hash(snapshot) != snapshot.audit_hash:
            return False
        previous = snapshot.audit_hash
    if result.final_audit_hash != previous:
        return False
    expected_completed = (
        len(result.snapshots) == result.input_snapshot_count and not result.failure_reasons
    )
    if result.completed != expected_completed or result.downstream_eligible != expected_completed:
        return False
    expected_provenance = bool(result.snapshots) and all(
        snapshot.input_store_snapshot_sha256 is not None and bool(snapshot.input_manifest_hashes)
        for snapshot in result.snapshots
    )
    if result.store_provenance_bound != expected_provenance:
        return False
    expected_estimated = bool(result.snapshots) and all(
        snapshot.input_estimate_sha256 is not None for snapshot in result.snapshots
    )
    if result.estimated_input_provenance_bound != expected_estimated:
        return False
    expected_spread = result.snapshots[-1].cumulative_spread_cost if result.snapshots else 0.0
    expected_cost = result.snapshots[-1].cumulative_transaction_cost if result.snapshots else 0.0
    expected_turnover = result.snapshots[-1].cumulative_turnover if result.snapshots else 0.0
    expected_gap = (
        max(abs(snapshot.accounting_gap) for snapshot in result.snapshots)
        if result.snapshots
        else None
    )
    if not math.isclose(result.total_spread_cost, expected_spread, rel_tol=0.0, abs_tol=1e-12):
        return False
    if not math.isclose(result.total_transaction_cost, expected_cost, rel_tol=0.0, abs_tol=1e-12):
        return False
    if not math.isclose(result.total_turnover, expected_turnover, rel_tol=0.0, abs_tol=1e-12):
        return False
    if result.maximum_absolute_accounting_gap != expected_gap:
        return False
    expected_pnl = result.snapshots[-1].portfolio_value if expected_completed else None
    return result.total_pnl == expected_pnl


def _terminal_payoff(
    terms: DeliverableTerms,
    market: BasketMarket,
    *,
    option_type: OptionType,
    strike: float,
) -> float:
    if terms.settlement_method == "cash":
        valuation = value_option_settlement(
            terms,
            option_type=option_type,
            strike=strike,
            settlement_level=market.assets[0].spot,
        )
    else:
        valuation = value_option_settlement(
            terms,
            option_type=option_type,
            strike=strike,
            security_marks={asset.asset_id: asset.spot for asset in market.assets},
        )
    return valuation.net_payoff


def run_basket_hedge_replay(
    frame: pd.DataFrame,
    terms: DeliverableTerms,
    *,
    option_type: str | OptionType,
    strike: float,
    expiry_timestamp: object,
    settings: BasketReplaySettings | None = None,
) -> BasketReplayResult:
    """Replay one short adjusted option with point-in-time component delta hedges."""
    settings = settings or BasketReplaySettings()
    kind = OptionType.parse(option_type)
    normalized_strike = _finite(strike, "strike")
    if normalized_strike < 0.0:
        raise ValueError("strike must be non-negative")
    expiry = _aware_timestamp(expiry_timestamp, "expiry_timestamp")
    missing = sorted(set(BASKET_REPLAY_COLUMNS) - set(frame.columns))
    if missing:
        raise BasketReplaySchemaError("missing basket replay columns: " + ", ".join(missing))
    if not 2 <= len(frame) <= MAX_BASKET_REPLAY_ROWS:
        raise BasketReplaySchemaError(
            f"basket replay requires between 2 and {MAX_BASKET_REPLAY_ROWS} rows"
        )
    supplied_provenance = set(BASKET_REPLAY_PROVENANCE_COLUMNS).intersection(frame.columns)
    if supplied_provenance and supplied_provenance != set(BASKET_REPLAY_PROVENANCE_COLUMNS):
        missing_provenance = sorted(set(BASKET_REPLAY_PROVENANCE_COLUMNS) - supplied_provenance)
        raise BasketReplaySchemaError(
            "incomplete basket-input provenance columns: " + ", ".join(missing_provenance)
        )
    has_store_provenance = bool(supplied_provenance)
    if settings.require_store_provenance and not has_store_provenance:
        raise BasketReplaySchemaError("basket replay requires append-only store provenance")
    supplied_estimation = set(BASKET_REPLAY_ESTIMATION_COLUMNS).intersection(frame.columns)
    if supplied_estimation and supplied_estimation != set(BASKET_REPLAY_ESTIMATION_COLUMNS):
        missing_estimation = sorted(set(BASKET_REPLAY_ESTIMATION_COLUMNS) - supplied_estimation)
        raise BasketReplaySchemaError(
            "incomplete estimated-input provenance columns: " + ", ".join(missing_estimation)
        )
    has_estimation_provenance = bool(supplied_estimation)
    if has_estimation_provenance and not has_store_provenance:
        raise BasketReplaySchemaError("estimated inputs require append-only store provenance")
    if settings.require_estimated_inputs and not has_estimation_provenance:
        raise BasketReplaySchemaError("basket replay requires statistically estimated inputs")
    formations = [
        _aware_timestamp(value, f"formation_timestamp row {index}")
        for index, value in enumerate(frame["formation_timestamp"])
    ]
    if any(later <= earlier for earlier, later in pairwise(formations)):
        raise BasketReplaySchemaError("formation timestamps must be strictly increasing")
    if any(timestamp > expiry for timestamp in formations) or formations[-1] != expiry:
        raise BasketReplaySchemaError("the final formation timestamp must equal expiry")
    if formations[0] >= expiry:
        raise BasketReplaySchemaError("basket replay requires a pre-expiry opening snapshot")
    configuration_hash = _configuration_hash(
        settings=settings,
        option_type=kind.value,
        strike=normalized_strike,
        expiry_timestamp=expiry.isoformat(),
        settlement_method=terms.settlement_method,
        classification=terms.classification,
        contract_multiplier=terms.contract_multiplier,
        currency=terms.currency,
        underlying_id=terms.underlying_id,
        deliverable_sha256=terms.sha256,
        input_snapshot_count=len(frame),
    )

    snapshots: list[BasketReplaySnapshotAudit] = []
    failures: list[str] = []
    positions: dict[str, float] = {}
    cash = 0.0
    cumulative_spread = 0.0
    cumulative_cost = 0.0
    cumulative_turnover = 0.0
    previous_value = 0.0
    previous_market: BasketMarket | None = None
    previous_rate = 0.0
    previous_formation: pd.Timestamp | None = None
    previous_audit_hash: str | None = None
    expected_asset_ids: tuple[str, ...] | None = None
    seen_source_records: set[tuple[str, str]] = set()

    for sequence, (_, row) in enumerate(frame.iterrows()):
        formation = formations[sequence]
        try:
            market_timestamp = _aware_timestamp(
                row["market_timestamp"], f"market_timestamp row {sequence}"
            )
            received = _aware_timestamp(
                row["received_timestamp"], f"received_timestamp row {sequence}"
            )
            raw_source = row["source"]
            raw_source_record_id = row["source_record_id"]
            if not isinstance(raw_source, str) or not raw_source.strip():
                raise ValueError("source must be a non-empty string")
            if not isinstance(raw_source_record_id, str) or not raw_source_record_id.strip():
                raise ValueError("source_record_id must be a non-empty string")
            source = raw_source.strip()
            source_record_id = raw_source_record_id.strip()
            source_key = (source, source_record_id)
            if source_key in seen_source_records:
                raise ValueError("source and source_record_id must be unique across snapshots")
            seen_source_records.add(source_key)
            if not market_timestamp <= received <= formation:
                raise ValueError(
                    "market_timestamp <= received_timestamp <= formation_timestamp is required"
                )
            market_age = (formation - market_timestamp).total_seconds()
            if market_age > settings.maximum_market_age_seconds:
                raise ValueError(
                    f"market package age {market_age:.6g}s exceeds "
                    f"{settings.maximum_market_age_seconds:.6g}s"
                )
            option_bid, option_ask, option_mid = _option_quote(row)
            rate = _finite(row["risk_free_rate"], "risk_free_rate")
            market = parse_basket_market(row["market_json"])
            execution_quotes, execution_json, execution_hash = _parse_execution_quotes(
                row["execution_json"], market
            )
            input_store_snapshot_hash, input_manifest_hashes = _store_provenance(
                row,
                supplied=has_store_provenance,
            )
            input_estimate_hash = _estimation_provenance(
                row,
                supplied=has_estimation_provenance,
                market=market,
                formation=formation,
                input_store_snapshot_sha256=input_store_snapshot_hash,
                input_manifest_hashes=input_manifest_hashes,
            )
            if expected_asset_ids is None:
                expected_asset_ids = market.asset_ids
            elif market.asset_ids != expected_asset_ids:
                raise ValueError("basket risk-factor identities cannot change during replay")
            maturity = (expiry - formation).total_seconds() / _SECONDS_PER_YEAR
            pricing = price_basket_option(
                terms,
                market,
                option_type=kind,
                strike=normalized_strike,
                maturity=maturity,
                rate=rate,
                settings=settings.monte_carlo(sequence),
            )
            if not pricing.bounds_passed:
                raise ArithmeticError("basket price failed its discounted bounds gate")
            is_final = formation == expiry
            settlement_payoff = (
                _terminal_payoff(
                    terms,
                    market,
                    option_type=kind,
                    strike=normalized_strike,
                )
                if is_final
                else None
            )
            if settlement_payoff is not None and (
                abs(option_bid - settlement_payoff) > settings.accounting_tolerance
                or abs(option_ask - settlement_payoff) > settings.accounting_tolerance
            ):
                raise ValueError(
                    "expiry option bid and ask must equal the explicit terminal settlement payoff"
                )

            cash_before = cash
            cash_carry = 0.0
            dividend_cashflow = 0.0
            if previous_formation is not None and previous_market is not None:
                elapsed = (formation - previous_formation).total_seconds() / _SECONDS_PER_YEAR
                cash_carry = cash * math.expm1(previous_rate * elapsed)
                previous_assets = {asset.asset_id: asset for asset in previous_market.assets}
                dividend_cashflow = sum(
                    positions.get(asset_id, 0.0)
                    * asset.spot
                    * math.expm1(asset.dividend_yield * elapsed)
                    for asset_id, asset in previous_assets.items()
                )
                cash += cash_carry + dividend_cashflow

            option_cashflow = 0.0
            option_spread = 0.0
            option_transaction_cost = 0.0
            if sequence == 0:
                option_cashflow = settings.short_option_contracts * option_bid
                option_spread = settings.short_option_contracts * (option_mid - option_bid)
                option_transaction_cost = (
                    settings.short_option_contracts
                    * option_bid
                    * settings.option_cost_bps
                    / 10_000.0
                )
            elif is_final:
                option_cashflow = -settings.short_option_contracts * float(settlement_payoff)

            delta_map = {delta.asset_id: delta for delta in pricing.deltas}
            target_positions = (
                {asset_id: 0.0 for asset_id in market.asset_ids}
                if is_final
                else {
                    asset_id: settings.short_option_contracts * delta_map[asset_id].delta
                    for asset_id in market.asset_ids
                }
            )
            component_rows: list[BasketComponentAudit] = []
            component_trade_cashflow = 0.0
            component_spread = 0.0
            component_transaction_cost = 0.0
            component_turnover = 0.0
            for asset, quote in zip(market.assets, execution_quotes, strict=True):
                prior = positions.get(asset.asset_id, 0.0)
                target = target_positions[asset.asset_id]
                trade = target - prior
                execution_price = (
                    quote.ask if trade > 0.0 else quote.bid if trade < 0.0 else asset.spot
                )
                trade_cashflow = -trade * execution_price
                spread_cost = -(-trade * execution_price + trade * asset.spot)
                transaction_cost = (
                    abs(trade * execution_price) * settings.component_cost_bps / 10_000.0
                )
                component_trade_cashflow += trade_cashflow
                component_spread += spread_cost
                component_transaction_cost += transaction_cost
                component_turnover += abs(trade * execution_price)
                delta = delta_map.get(asset.asset_id)
                component_rows.append(
                    BasketComponentAudit(
                        asset_id=asset.asset_id,
                        spot=asset.spot,
                        bid=quote.bid,
                        ask=quote.ask,
                        model_delta=delta.delta if delta is not None else 0.0,
                        delta_standard_error=delta.standard_error if delta is not None else 0.0,
                        prior_position=prior,
                        target_position=target,
                        trade=trade,
                        execution_price=execution_price,
                        trade_cashflow=trade_cashflow,
                        spread_cost=spread_cost,
                        transaction_cost=transaction_cost,
                    )
                )

            current_option_mark = (
                settings.short_option_contracts * float(settlement_payoff)
                if is_final
                else settings.short_option_contracts * option_mid
            )
            pretrade_value = (
                cash
                + sum(positions.get(asset.asset_id, 0.0) * asset.spot for asset in market.assets)
                - (0.0 if sequence == 0 else current_option_mark)
            )
            spread_cost = option_spread + component_spread
            transaction_cost = option_transaction_cost + component_transaction_cost
            turnover = component_turnover + (
                settings.short_option_contracts * option_bid if sequence == 0 else 0.0
            )
            cash += option_cashflow + component_trade_cashflow - transaction_cost
            positions = target_positions
            portfolio_value = (
                cash
                + sum(positions.get(asset.asset_id, 0.0) * asset.spot for asset in market.assets)
                - (0.0 if is_final else current_option_mark)
            )
            accounting_gap = portfolio_value - (pretrade_value - spread_cost - transaction_cost)
            if abs(accounting_gap) > settings.accounting_tolerance:
                raise ArithmeticError(
                    f"self-financing accounting gap {accounting_gap:.6g} exceeds tolerance"
                )

            cumulative_spread += spread_cost
            cumulative_cost += transaction_cost
            cumulative_turnover += turnover
            incremental_pnl = portfolio_value - previous_value
            snapshot_hash = _snapshot_input_hash(
                formation=formation,
                market_timestamp=market_timestamp,
                received=received,
                source=source,
                source_record_id=source_record_id,
                option_bid=option_bid,
                option_ask=option_ask,
                rate=rate,
                market=market,
                execution_json=execution_json,
                input_store_snapshot_sha256=input_store_snapshot_hash,
                input_manifest_hashes=input_manifest_hashes,
                input_estimate_sha256=input_estimate_hash,
            )
            snapshot = BasketReplaySnapshotAudit(
                sequence=sequence,
                formation_timestamp=formation.isoformat(),
                market_timestamp=market_timestamp.isoformat(),
                received_timestamp=received.isoformat(),
                action="settle" if is_final else "open" if sequence == 0 else "rebalance",
                source=source,
                source_record_id=source_record_id,
                snapshot_sha256=snapshot_hash,
                market_sha256=market.sha256,
                execution_sha256=execution_hash,
                input_store_snapshot_sha256=input_store_snapshot_hash,
                input_manifest_hashes=input_manifest_hashes,
                input_estimate_sha256=input_estimate_hash,
                configuration_sha256=configuration_hash,
                previous_audit_hash=previous_audit_hash,
                audit_hash="",
                no_lookahead_passed=True,
                market_age_seconds=market_age,
                maturity_years=maturity,
                rate=rate,
                option_bid=option_bid,
                option_ask=option_ask,
                option_mid=option_mid,
                option_model_price=pricing.price,
                option_model_standard_error=pricing.standard_error,
                option_model_error_to_mid=pricing.price - option_mid,
                pricing_bounds_passed=pricing.bounds_passed,
                settlement_payoff=settlement_payoff,
                option_cashflow=option_cashflow,
                cash_before=cash_before,
                cash_carry=cash_carry,
                dividend_cashflow=dividend_cashflow,
                component_trade_cashflow=component_trade_cashflow,
                spread_cost=spread_cost,
                cumulative_spread_cost=cumulative_spread,
                transaction_cost=transaction_cost,
                cumulative_transaction_cost=cumulative_cost,
                turnover=turnover,
                cumulative_turnover=cumulative_turnover,
                cash=cash,
                portfolio_value=portfolio_value,
                incremental_pnl=incremental_pnl,
                cumulative_pnl=portfolio_value,
                accounting_gap=accounting_gap,
                components=tuple(component_rows),
            )
            snapshot = replace(snapshot, audit_hash=_audit_hash(snapshot))
            snapshots.append(snapshot)
            previous_audit_hash = snapshot.audit_hash
            previous_value = portfolio_value
            previous_market = market
            previous_rate = rate
            previous_formation = formation
        except (ArithmeticError, BasketReplaySchemaError, ValueError) as exc:
            failures.append(f"snapshot {sequence} ({formation.isoformat()}): {exc}")
            break

    completed = len(snapshots) == len(frame) and not failures
    gaps = [abs(snapshot.accounting_gap) for snapshot in snapshots]
    result = BasketReplayResult(
        settings=settings,
        option_type=kind.value,
        strike=normalized_strike,
        expiry_timestamp=expiry.isoformat(),
        settlement_method=terms.settlement_method,
        classification=terms.classification,
        contract_multiplier=terms.contract_multiplier,
        currency=terms.currency,
        underlying_id=terms.underlying_id,
        deliverable_sha256=terms.sha256,
        configuration_sha256=configuration_hash,
        input_snapshot_count=len(frame),
        snapshots=tuple(snapshots),
        completed=completed,
        downstream_eligible=completed,
        store_provenance_bound=bool(snapshots)
        and all(
            snapshot.input_store_snapshot_sha256 is not None
            and bool(snapshot.input_manifest_hashes)
            for snapshot in snapshots
        ),
        estimated_input_provenance_bound=bool(snapshots)
        and all(snapshot.input_estimate_sha256 is not None for snapshot in snapshots),
        total_pnl=snapshots[-1].portfolio_value if completed else None,
        total_spread_cost=cumulative_spread,
        total_transaction_cost=cumulative_cost,
        total_turnover=cumulative_turnover,
        maximum_absolute_accounting_gap=max(gaps) if gaps else None,
        final_audit_hash=previous_audit_hash,
        failure_reasons=tuple(failures),
    )
    if not verify_basket_replay_hash_chain(result):
        return replace(
            result,
            downstream_eligible=False,
            failure_reasons=(
                *result.failure_reasons,
                "basket replay audit chain failed verification",
            ),
        )
    return result


def basket_replay_report_json(result: BasketReplayResult) -> bytes:
    """Serialize a deterministic human-readable basket replay audit."""
    return (json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")


def generate_synthetic_basket_replay(
    *,
    paths: int = 4_000,
    seed: int = 101,
) -> pd.DataFrame:
    """Generate a deterministic adjusted-option replay ending at explicit settlement."""
    terms = parse_deliverable_terms(
        SYNTHETIC_BASKET_DELIVERABLE,
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ACME",
    )
    expiry = _aware_timestamp(SYNTHETIC_BASKET_EXPIRY, "synthetic expiry")
    timestamps = (
        pd.Timestamp("2026-04-01T20:00:00Z"),
        pd.Timestamp("2026-04-10T20:00:00Z"),
        pd.Timestamp("2026-04-20T20:00:00Z"),
        expiry,
    )
    spots = ((120.0, 20.0), (122.0, 21.0), (119.0, 20.5), (125.0, 22.0))
    volatilities = ((0.25, 0.35), (0.24, 0.33), (0.22, 0.30), (0.20, 0.28))
    correlations = (0.35, 0.34, 0.32, 0.30)
    rows: list[dict[str, object]] = []
    for sequence, (formation, spot_pair, vol_pair, correlation) in enumerate(
        zip(timestamps, spots, volatilities, correlations, strict=True)
    ):
        market_payload = {
            "assets": [
                {
                    "asset_id": "ACME",
                    "spot": spot_pair[0],
                    "volatility": vol_pair[0],
                    "dividend_yield": 0.01,
                },
                {
                    "asset_id": "SPIN",
                    "spot": spot_pair[1],
                    "volatility": vol_pair[1],
                    "dividend_yield": 0.0,
                },
            ],
            "correlation": [[1.0, correlation], [correlation, 1.0]],
        }
        market = parse_basket_market(market_payload)
        maturity = (expiry - formation).total_seconds() / _SECONDS_PER_YEAR
        if maturity == 0.0:
            fair_value = _terminal_payoff(
                terms,
                market,
                option_type=OptionType.CALL,
                strike=60.0,
            )
            option_bid = fair_value
            option_ask = fair_value
        else:
            pricing = price_basket_option(
                terms,
                market,
                option_type="call",
                strike=60.0,
                maturity=maturity,
                rate=0.03,
                settings=BasketMonteCarloSettings(paths=paths, seed=seed + sequence),
            )
            half_spread = 1.0
            option_bid = max(pricing.price - half_spread, 0.0)
            option_ask = pricing.price + half_spread
        execution = [
            {"asset_id": asset.asset_id, "bid": asset.spot - 0.02, "ask": asset.spot + 0.02}
            for asset in market.assets
        ]
        market_timestamp = formation - pd.Timedelta(minutes=2)
        received = formation - pd.Timedelta(minutes=1)
        rows.append(
            {
                "formation_timestamp": formation.isoformat(),
                "market_timestamp": market_timestamp.isoformat(),
                "received_timestamp": received.isoformat(),
                "source": "volforge-synthetic-v1",
                "source_record_id": f"SYN-BASKET-{sequence + 1:04d}",
                "option_bid": option_bid,
                "option_ask": option_ask,
                "risk_free_rate": 0.03,
                "market_json": market.canonical_json,
                "execution_json": _canonical_json(execution),
            }
        )
    return pd.DataFrame(rows, columns=BASKET_REPLAY_COLUMNS)


__all__ = [
    "BASKET_REPLAY_COLUMNS",
    "BASKET_REPLAY_ESTIMATION_COLUMNS",
    "BASKET_REPLAY_PROVENANCE_COLUMNS",
    "MAX_BASKET_REPLAY_ROWS",
    "SYNTHETIC_BASKET_DELIVERABLE",
    "SYNTHETIC_BASKET_EXPIRY",
    "BasketComponentAudit",
    "BasketReplayResult",
    "BasketReplaySchemaError",
    "BasketReplaySettings",
    "BasketReplaySnapshotAudit",
    "ComponentExecutionQuote",
    "basket_replay_report_json",
    "generate_synthetic_basket_replay",
    "run_basket_hedge_replay",
    "verify_basket_replay_hash_chain",
]
