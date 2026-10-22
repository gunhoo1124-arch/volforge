"""Deterministic listed-option deliverable parsing and settlement accounting."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from volforge.models.common import OptionType

DELIVERABLE_SCHEMA_VERSION = 1
MAX_DELIVERABLE_BYTES = 1_048_576
MAX_DELIVERABLE_COMPONENTS = 64
SUPPORTED_ASSET_TYPES = ("security", "cash", "cash_in_lieu", "cash_settlement")

_CURRENCY_PATTERN = re.compile(r"[A-Z]{3}")
_COMPONENT_FIELDS = {"asset_id", "asset_type", "quantity", "currency", "description"}


@dataclass(frozen=True, slots=True)
class DeliverableComponent:
    """One normalized component of an option contract deliverable."""

    asset_id: str
    asset_type: str
    quantity: float
    currency: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class DeliverableTerms:
    """Validated, canonical contract deliverable terms."""

    settlement_method: str
    classification: str
    contract_multiplier: float
    currency: str
    underlying_id: str | None
    components: tuple[DeliverableComponent, ...]
    canonical_json: str
    sha256: str
    required_security_marks: tuple[str, ...]
    single_factor_pricing_compatible: bool


@dataclass(frozen=True, slots=True)
class ComponentValuation:
    """Auditable value assigned to one deliverable component."""

    asset_id: str
    asset_type: str
    quantity: float
    currency: str
    unit_mark: float
    gross_value: float
    description: str | None = None


@dataclass(frozen=True, slots=True)
class SettlementValuation:
    """Long-option settlement cash flows for one listed contract."""

    schema_version: int
    deliverable_sha256: str
    classification: str
    settlement_method: str
    option_type: str
    currency: str
    contract_multiplier: float
    strike: float
    settlement_level: float | None
    aggregate_exercise_price: float
    delivered_value: float
    gross_receivable: float
    gross_payable: float
    net_payoff: float
    in_the_money: bool
    components: tuple[ComponentValuation, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready audit payload."""
        payload = asdict(self)
        payload["components"] = [asdict(component) for component in self.components]
        return payload


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _currency(value: object, name: str) -> str:
    currency = str(value).strip().upper()
    if not _CURRENCY_PATTERN.fullmatch(currency):
        raise ValueError(f"{name} must be a three-letter currency code")
    return currency


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _json_value(value: object, name: str) -> Any:
    if not isinstance(value, str):
        return value
    if len(value.encode("utf-8")) > MAX_DELIVERABLE_BYTES:
        raise ValueError(f"{name} exceeds {MAX_DELIVERABLE_BYTES} bytes")
    try:
        return json.loads(value, object_pairs_hook=_strict_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON") from exc


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def parse_deliverable_terms(
    deliverable: object,
    *,
    settlement_method: str,
    contract_multiplier: float,
    currency: str,
    underlying_id: str | None = None,
) -> DeliverableTerms:
    """Validate and canonicalize listed-option deliverable terms.

    A missing physical deliverable is interpreted as the standard single-security
    contract used by legacy, unreferenced datasets. Cash-settled contracts must
    always declare one ``cash_settlement`` component explicitly.
    """
    method = str(settlement_method).strip().lower()
    if method not in {"physical", "cash"}:
        raise ValueError("settlement_method must be 'physical' or 'cash'")
    multiplier = _finite_number(contract_multiplier, "contract_multiplier", positive=True)
    settlement_currency = _currency(currency, "currency")
    normalized_underlying = str(underlying_id).strip() if underlying_id is not None else None
    if normalized_underlying == "":
        raise ValueError("underlying_id must be non-empty when supplied")

    if deliverable is None:
        if method != "physical":
            raise ValueError("cash settlement requires an explicit cash_settlement component")
        deliverable = [
            {
                "asset_id": normalized_underlying or "__underlying__",
                "asset_type": "security",
                "quantity": multiplier,
                "currency": settlement_currency,
                "description": "standard unreferenced deliverable",
            }
        ]
    parsed = _json_value(deliverable, "deliverable_json")
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("deliverable_json must be a non-empty JSON list")
    if len(parsed) > MAX_DELIVERABLE_COMPONENTS:
        raise ValueError(f"deliverable_json exceeds {MAX_DELIVERABLE_COMPONENTS} components")

    components: list[DeliverableComponent] = []
    identities: set[tuple[str, str]] = set()
    for position, raw_component in enumerate(parsed):
        prefix = f"deliverable component {position}"
        if not isinstance(raw_component, dict):
            raise ValueError(f"{prefix} must be an object")
        unexpected = sorted(set(raw_component) - _COMPONENT_FIELDS)
        if unexpected:
            raise ValueError(f"{prefix} has unsupported fields: {', '.join(unexpected)}")
        asset_id = str(raw_component.get("asset_id", "")).strip()
        if not asset_id:
            raise ValueError(f"{prefix} asset_id must be non-empty")
        asset_type = str(raw_component.get("asset_type", "")).strip().lower()
        if asset_type not in SUPPORTED_ASSET_TYPES:
            raise ValueError(
                f"{prefix} asset_type must be one of {', '.join(SUPPORTED_ASSET_TYPES)}"
            )
        quantity = _finite_number(
            raw_component.get("quantity"), f"{prefix} quantity", positive=True
        )
        component_currency = _currency(
            raw_component.get("currency", settlement_currency), f"{prefix} currency"
        )
        if component_currency != settlement_currency:
            raise ValueError(
                f"{prefix} currency differs from settlement currency; supply an explicit "
                "converted amount or mark"
            )
        raw_description = raw_component.get("description")
        description = None if raw_description is None else str(raw_description).strip() or None
        identity = (asset_type, asset_id)
        if identity in identities:
            raise ValueError(f"duplicate deliverable component {asset_type}:{asset_id}")
        identities.add(identity)
        components.append(
            DeliverableComponent(
                asset_id=asset_id,
                asset_type=asset_type,
                quantity=quantity,
                currency=component_currency,
                description=description,
            )
        )

    components.sort(
        key=lambda component: (
            component.asset_type,
            component.asset_id,
            component.currency,
            component.quantity,
            component.description or "",
        )
    )
    cash_settlement_components = [
        component for component in components if component.asset_type == "cash_settlement"
    ]
    if method == "cash":
        if len(components) != 1 or len(cash_settlement_components) != 1:
            raise ValueError("cash settlement requires exactly one cash_settlement component")
        if not math.isclose(
            cash_settlement_components[0].quantity,
            multiplier,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError("cash_settlement quantity must equal contract_multiplier")
        classification = "cash_settled"
        compatible = False
    else:
        if cash_settlement_components:
            raise ValueError("physical settlement cannot contain cash_settlement components")
        standard = (
            len(components) == 1
            and components[0].asset_type == "security"
            and math.isclose(components[0].quantity, multiplier, rel_tol=0.0, abs_tol=1e-10)
            and (normalized_underlying is None or components[0].asset_id == normalized_underlying)
        )
        classification = "standard_physical" if standard else "adjusted_physical"
        compatible = standard

    canonical_components = [asdict(component) for component in components]
    canonical = _canonical_json(canonical_components)
    required_marks = tuple(
        component.asset_id for component in components if component.asset_type == "security"
    )
    return DeliverableTerms(
        settlement_method=method,
        classification=classification,
        contract_multiplier=multiplier,
        currency=settlement_currency,
        underlying_id=normalized_underlying,
        components=tuple(components),
        canonical_json=canonical,
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        required_security_marks=required_marks,
        single_factor_pricing_compatible=compatible,
    )


def parse_security_marks(value: object | None) -> dict[str, float]:
    """Parse a strict JSON object mapping security identifiers to unit marks."""
    if value is None:
        return {}
    parsed = _json_value(value, "security_marks")
    if not isinstance(parsed, Mapping):
        raise ValueError("security_marks must be a JSON object")
    marks: dict[str, float] = {}
    for raw_asset_id, raw_mark in parsed.items():
        asset_id = str(raw_asset_id).strip()
        if not asset_id:
            raise ValueError("security_marks asset identifiers must be non-empty")
        mark = _finite_number(raw_mark, f"security mark for {asset_id}")
        if mark < 0.0:
            raise ValueError(f"security mark for {asset_id} must be non-negative")
        marks[asset_id] = mark
    return marks


def value_option_settlement(
    terms: DeliverableTerms,
    *,
    option_type: str | OptionType,
    strike: float,
    security_marks: object | None = None,
    underlying_mark: float | None = None,
    settlement_level: float | None = None,
) -> SettlementValuation:
    """Value the long-option settlement cash flows for one listed contract."""
    kind = OptionType.parse(option_type)
    normalized_strike = _finite_number(strike, "strike")
    if normalized_strike < 0.0:
        raise ValueError("strike must be non-negative")
    marks = parse_security_marks(security_marks)

    if underlying_mark is not None:
        mark = _finite_number(underlying_mark, "underlying_mark")
        if mark < 0.0:
            raise ValueError("underlying_mark must be non-negative")
        mark_asset_id = terms.underlying_id
        if mark_asset_id is None and terms.classification == "standard_physical":
            mark_asset_id = terms.components[0].asset_id
        if mark_asset_id is None:
            raise ValueError("underlying_mark requires an underlying_id")
        if mark_asset_id in marks and not math.isclose(
            marks[mark_asset_id], mark, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("underlying_mark conflicts with security_marks")
        marks[mark_asset_id] = mark

    required_marks = set(terms.required_security_marks)
    missing_marks = sorted(required_marks - set(marks))
    if missing_marks:
        raise ValueError("missing security marks: " + ", ".join(missing_marks))
    unused_marks = sorted(set(marks) - required_marks)
    if unused_marks:
        raise ValueError("unused security marks: " + ", ".join(unused_marks))

    aggregate_exercise_price = normalized_strike * terms.contract_multiplier
    component_values: list[ComponentValuation] = []
    if terms.settlement_method == "cash":
        if settlement_level is None:
            raise ValueError("cash settlement requires settlement_level")
        level = _finite_number(settlement_level, "settlement_level")
        if level < 0.0:
            raise ValueError("settlement_level must be non-negative")
        component = terms.components[0]
        delivered_value = component.quantity * level
        component_values.append(
            ComponentValuation(
                asset_id=component.asset_id,
                asset_type=component.asset_type,
                quantity=component.quantity,
                currency=component.currency,
                unit_mark=level,
                gross_value=delivered_value,
                description=component.description,
            )
        )
        intrinsic = (
            delivered_value - aggregate_exercise_price
            if kind is OptionType.CALL
            else aggregate_exercise_price - delivered_value
        )
        payoff = max(intrinsic, 0.0)
        receivable = payoff
        payable = 0.0
    else:
        if settlement_level is not None:
            raise ValueError("settlement_level applies only to cash-settled contracts")
        level = None
        delivered_value = 0.0
        for component in terms.components:
            unit_mark = marks[component.asset_id] if component.asset_type == "security" else 1.0
            gross_value = component.quantity * unit_mark
            delivered_value += gross_value
            component_values.append(
                ComponentValuation(
                    asset_id=component.asset_id,
                    asset_type=component.asset_type,
                    quantity=component.quantity,
                    currency=component.currency,
                    unit_mark=unit_mark,
                    gross_value=gross_value,
                    description=component.description,
                )
            )
        intrinsic = (
            delivered_value - aggregate_exercise_price
            if kind is OptionType.CALL
            else aggregate_exercise_price - delivered_value
        )
        payoff = max(intrinsic, 0.0)
        if payoff == 0.0:
            receivable = 0.0
            payable = 0.0
        elif kind is OptionType.CALL:
            receivable = delivered_value
            payable = aggregate_exercise_price
        else:
            receivable = aggregate_exercise_price
            payable = delivered_value

    return SettlementValuation(
        schema_version=DELIVERABLE_SCHEMA_VERSION,
        deliverable_sha256=terms.sha256,
        classification=terms.classification,
        settlement_method=terms.settlement_method,
        option_type=kind.value,
        currency=terms.currency,
        contract_multiplier=terms.contract_multiplier,
        strike=normalized_strike,
        settlement_level=level,
        aggregate_exercise_price=aggregate_exercise_price,
        delivered_value=delivered_value,
        gross_receivable=receivable,
        gross_payable=payable,
        net_payoff=payoff,
        in_the_money=payoff > 0.0,
        components=tuple(component_values),
    )


__all__ = [
    "DELIVERABLE_SCHEMA_VERSION",
    "MAX_DELIVERABLE_BYTES",
    "MAX_DELIVERABLE_COMPONENTS",
    "SUPPORTED_ASSET_TYPES",
    "ComponentValuation",
    "DeliverableComponent",
    "DeliverableTerms",
    "SettlementValuation",
    "parse_deliverable_terms",
    "parse_security_marks",
    "value_option_settlement",
]
