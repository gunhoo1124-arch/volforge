"""Deterministic synthetic basket-option hedge fixtures."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from volforge.basket import (
    BasketMonteCarloSettings,
    parse_basket_market,
    price_basket_option,
)
from volforge.deliverables import parse_deliverable_terms

_SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _timestamp(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid timestamp")
    return pd.Timestamp(timestamp)


_SYNTHETIC_HEDGE_DEFINITIONS = (
    {
        "instrument_id": "GAMMA-CALL-202606",
        "option_type": "call",
        "strike": 67.0,
        "expiry_timestamp": "2026-06-30T20:00:00+00:00",
        "deliverable_json": [
            {"asset_id": "ACME", "asset_type": "security", "currency": "USD", "quantity": 55},
            {"asset_id": "SPIN", "asset_type": "security", "currency": "USD", "quantity": 5},
            {
                "asset_id": "USD-CIL",
                "asset_type": "cash_in_lieu",
                "currency": "USD",
                "quantity": 20,
            },
        ],
    },
    {
        "instrument_id": "VEGA-CALL-202607",
        "option_type": "call",
        "strike": 34.0,
        "expiry_timestamp": "2026-07-31T20:00:00+00:00",
        "deliverable_json": [
            {"asset_id": "ACME", "asset_type": "security", "currency": "USD", "quantity": 20},
            {"asset_id": "SPIN", "asset_type": "security", "currency": "USD", "quantity": 45},
            {
                "asset_id": "USD-CIL",
                "asset_type": "cash_in_lieu",
                "currency": "USD",
                "quantity": 20,
            },
        ],
    },
    {
        "instrument_id": "CORR-CALL-202609",
        "option_type": "call",
        "strike": 50.0,
        "expiry_timestamp": "2026-09-30T20:00:00+00:00",
        "deliverable_json": [
            {"asset_id": "ACME", "asset_type": "security", "currency": "USD", "quantity": 35},
            {"asset_id": "SPIN", "asset_type": "security", "currency": "USD", "quantity": 35},
            {
                "asset_id": "USD-CIL",
                "asset_type": "cash_in_lieu",
                "currency": "USD",
                "quantity": 20,
            },
        ],
    },
)

_SYNTHETIC_SMILE_HEDGE_DEFINITIONS = (
    {
        "instrument_id": "LOW-WING-PUT-202606",
        "option_type": "put",
        "strike": 55.0,
        "expiry_timestamp": "2026-06-30T20:00:00+00:00",
        "deliverable_json": [
            {"asset_id": "ACME", "asset_type": "security", "currency": "USD", "quantity": 55},
            {"asset_id": "SPIN", "asset_type": "security", "currency": "USD", "quantity": 5},
            {
                "asset_id": "USD-CIL",
                "asset_type": "cash_in_lieu",
                "currency": "USD",
                "quantity": 20,
            },
        ],
    },
    _SYNTHETIC_HEDGE_DEFINITIONS[1],
    _SYNTHETIC_HEDGE_DEFINITIONS[2],
    {
        "instrument_id": "DEEP-LOW-PUT-202610",
        "option_type": "put",
        "strike": 43.0,
        "expiry_timestamp": "2026-10-30T20:00:00+00:00",
        "deliverable_json": [
            {"asset_id": "ACME", "asset_type": "security", "currency": "USD", "quantity": 45},
            {"asset_id": "SPIN", "asset_type": "security", "currency": "USD", "quantity": 15},
            {
                "asset_id": "USD-CIL",
                "asset_type": "cash_in_lieu",
                "currency": "USD",
                "quantity": 20,
            },
        ],
    },
    {
        "instrument_id": "HIGH-WING-CALL-202612",
        "option_type": "call",
        "strike": 42.0,
        "expiry_timestamp": "2026-12-31T20:00:00+00:00",
        "deliverable_json": [
            {"asset_id": "ACME", "asset_type": "security", "currency": "USD", "quantity": 25},
            {"asset_id": "SPIN", "asset_type": "security", "currency": "USD", "quantity": 30},
            {
                "asset_id": "USD-CIL",
                "asset_type": "cash_in_lieu",
                "currency": "USD",
                "quantity": 20,
            },
        ],
    },
)


def _synthetic_instruments_json(
    definitions: tuple[dict[str, Any], ...],
    market_value: object,
    *,
    formation_timestamp: str | pd.Timestamp,
    rate: float,
    paths: int,
) -> str:
    market = parse_basket_market(market_value)
    formation = _timestamp(formation_timestamp, "formation_timestamp")
    payload: list[dict[str, Any]] = []
    for index, definition in enumerate(definitions):
        terms = parse_deliverable_terms(
            definition["deliverable_json"],
            settlement_method="physical",
            contract_multiplier=100,
            currency="USD",
            underlying_id="ACME",
        )
        expiry = _timestamp(definition["expiry_timestamp"], "instrument expiry")
        maturity = (expiry - formation).total_seconds() / _SECONDS_PER_YEAR
        pricing = price_basket_option(
            terms,
            market,
            option_type=str(definition["option_type"]),
            strike=float(definition["strike"]),
            maturity=maturity,
            rate=rate,
            settings=BasketMonteCarloSettings(paths=paths, seed=900 + index),
        )
        half_spread = max(0.25, 0.0025 * pricing.price)
        payload.append(
            {
                **definition,
                "bid": max(pricing.price - half_spread, 0.0),
                "ask": pricing.price + half_spread,
                "settlement_method": "physical",
                "contract_multiplier": 100,
                "currency": "USD",
                "underlying_id": "ACME",
            }
        )
    return _canonical_json(payload)


def synthetic_hedge_instruments_json(
    market_value: object,
    *,
    formation_timestamp: str | pd.Timestamp,
    rate: float,
    paths: int = 8_000,
) -> str:
    """Quote the three deterministic synthetic hedge options around model mid."""
    return _synthetic_instruments_json(
        _SYNTHETIC_HEDGE_DEFINITIONS,
        market_value,
        formation_timestamp=formation_timestamp,
        rate=rate,
        paths=paths,
    )


def synthetic_smile_hedge_instruments_json(
    market_value: object,
    *,
    formation_timestamp: str | pd.Timestamp,
    rate: float,
    paths: int = 8_000,
) -> str:
    """Quote five deterministic instruments spanning level, smile, and correlation risks."""
    return _synthetic_instruments_json(
        _SYNTHETIC_SMILE_HEDGE_DEFINITIONS,
        market_value,
        formation_timestamp=formation_timestamp,
        rate=rate,
        paths=paths,
    )


__all__ = [
    "synthetic_hedge_instruments_json",
    "synthetic_smile_hedge_instruments_json",
]
