from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volforge.basket_estimation import (
    BasketEstimationSettings,
    estimate_basket_market,
    generate_synthetic_statistical_basket_inputs,
    materialize_estimated_replay,
    verify_basket_input_estimate,
)
from volforge.basket_inputs import (
    SYNTHETIC_BASKET_ID,
    BasketInputIntegrityError,
    BasketInputStore,
)
from volforge.basket_replay import (
    SYNTHETIC_BASKET_DELIVERABLE,
    SYNTHETIC_BASKET_EXPIRY,
    BasketReplaySettings,
    run_basket_hedge_replay,
)
from volforge.deliverables import parse_deliverable_terms

SOURCE = "volforge-synthetic-statistical-v1"


def _store(path: Path) -> tuple[BasketInputStore, pd.DataFrame]:
    records, schedule = generate_synthetic_statistical_basket_inputs()
    store = BasketInputStore.create(path)
    store.ingest_frame(
        records,
        source_file_name="statistical.csv",
        source=SOURCE,
        mode="initial",
        ingested_timestamp="2026-05-01T00:00:00Z",
    )
    return store, schedule


def test_estimate_is_deterministic_shrunk_psd_and_no_lookahead(tmp_path: Path) -> None:
    store, schedule = _store(tmp_path / "basket.sqlite")
    formation = schedule.iloc[0]["formation_timestamp"]
    first = estimate_basket_market(
        store,
        SYNTHETIC_BASKET_ID,
        formation,
        source=SOURCE,
    )
    second = estimate_basket_market(
        store,
        SYNTHETIC_BASKET_ID,
        formation,
        source=SOURCE,
    )

    assert first == second
    assert first.return_observation_count == 20
    assert first.no_lookahead_passed
    assert pd.Timestamp(first.observation_end) <= pd.Timestamp(first.formation_timestamp)
    assert first.shrunk_minimum_eigenvalue > first.sample_minimum_eigenvalue
    assert np.linalg.eigvalsh(first.market.correlation).min() >= 0.0
    assert verify_basket_input_estimate(first)
    assert verify_basket_input_estimate(first.canonical_json)
    tampered = json.loads(first.canonical_json)
    tampered["annualized_volatilities"][0] += 0.01
    assert not verify_basket_input_estimate(tampered)


def test_materialized_estimates_bind_and_feed_legacy_delta_replay(tmp_path: Path) -> None:
    store, schedule = _store(tmp_path / "basket.sqlite")
    frame = materialize_estimated_replay(
        store,
        schedule,
        basket_id=SYNTHETIC_BASKET_ID,
        source=SOURCE,
    )
    terms = parse_deliverable_terms(
        SYNTHETIC_BASKET_DELIVERABLE,
        settlement_method="physical",
        contract_multiplier=100,
        currency="USD",
        underlying_id="ACME",
    )
    result = run_basket_hedge_replay(
        frame,
        terms,
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=BasketReplaySettings(
            paths=2_000,
            require_store_provenance=True,
            require_estimated_inputs=True,
        ),
    )

    assert result.downstream_eligible
    assert result.store_provenance_bound
    assert result.estimated_input_provenance_bound
    assert all(snapshot.input_estimate_sha256 for snapshot in result.snapshots)

    broken = frame.copy()
    payload = json.loads(str(broken.loc[0, "input_estimate_json"]))
    payload["return_observation_count"] += 1
    broken.loc[0, "input_estimate_json"] = json.dumps(payload)
    blocked = run_basket_hedge_replay(
        broken,
        terms,
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=BasketReplaySettings(paths=2_000, require_estimated_inputs=True),
    )
    assert not blocked.downstream_eligible
    assert "self-hash" in blocked.failure_reasons[0]


def test_estimation_rejects_insufficient_history_and_bad_settings(tmp_path: Path) -> None:
    records, schedule = generate_synthetic_statistical_basket_inputs()
    cutoff = pd.Timestamp(schedule.iloc[0]["formation_timestamp"]) - pd.Timedelta(days=5)
    recent = records.loc[pd.to_datetime(records["market_timestamp"], utc=True) >= cutoff].copy()
    store = BasketInputStore.create(tmp_path / "short.sqlite")
    store.ingest_frame(
        recent,
        source_file_name="short.csv",
        source=SOURCE,
        mode="initial",
        ingested_timestamp="2026-05-01T00:00:00Z",
    )
    with pytest.raises(BasketInputIntegrityError, match="at least 10 synchronized returns"):
        estimate_basket_market(
            store,
            SYNTHETIC_BASKET_ID,
            schedule.iloc[0]["formation_timestamp"],
            source=SOURCE,
        )
    with pytest.raises(ValueError, match="window_returns"):
        BasketEstimationSettings(window_returns=5, minimum_returns=10)
    with pytest.raises(ValueError, match="correlation_shrinkage"):
        BasketEstimationSettings(correlation_shrinkage=1.1)
