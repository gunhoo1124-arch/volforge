from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volforge.basket_covariance import (
    BasketCovarianceSelectionSettings,
    _forecast_scores,
    basket_covariance_selection_report_json,
    materialize_selected_covariance_replay,
    select_basket_covariance,
    verify_basket_covariance_selection,
)
from volforge.basket_estimation import (
    COVARIANCE_ESTIMATORS,
    BasketEstimationSettings,
    estimate_basket_market,
    generate_synthetic_statistical_basket_inputs,
)
from volforge.basket_inputs import (
    SYNTHETIC_BASKET_ID,
    BasketInputIntegrityError,
    BasketInputStore,
)
from volforge.basket_replay import SYNTHETIC_BASKET_DELIVERABLE, SYNTHETIC_BASKET_EXPIRY
from volforge.basket_risk_hedging import (
    BasketRiskHedgeSchemaError,
    BasketRiskHedgeSettings,
    run_basket_risk_hedge_replay,
)
from volforge.deliverables import parse_deliverable_terms

SOURCE = "volforge-synthetic-statistical-v1"


def test_student_t_score_has_gaussian_limit_and_downweights_outliers() -> None:
    covariance = np.asarray([[0.04, 0.01], [0.01, 0.09]])
    mean = np.zeros(2)
    evaluation = np.asarray([0.10, -0.20])
    gaussian, limiting_student_t = _forecast_scores(
        evaluation,
        mean,
        covariance,
        student_t_degrees_of_freedom=1e6,
    )
    residual = evaluation - mean
    expected = 0.5 * (
        2.0 * math.log(2.0 * math.pi)
        + np.linalg.slogdet(covariance)[1]
        + residual @ np.linalg.solve(covariance, residual)
    )
    assert gaussian == pytest.approx(expected)
    assert limiting_student_t == pytest.approx(gaussian, abs=1e-6)

    outlier = np.asarray([1.0, -1.0])
    gaussian_outlier, student_t_outlier = _forecast_scores(
        outlier,
        mean,
        covariance,
        student_t_degrees_of_freedom=6.0,
    )
    assert student_t_outlier < gaussian_outlier


@pytest.fixture
def covariance_store(tmp_path: Path) -> tuple[BasketInputStore, pd.DataFrame]:
    records, schedule = generate_synthetic_statistical_basket_inputs()
    store = BasketInputStore.create(tmp_path / "basket.sqlite")
    store.ingest_frame(
        records,
        source_file_name="statistical.csv",
        source=SOURCE,
        mode="initial",
        ingested_timestamp="2026-05-01T00:00:00Z",
    )
    return store, schedule


def test_selection_is_chronological_deterministic_stable_and_hashed(
    covariance_store: tuple[BasketInputStore, pd.DataFrame],
) -> None:
    store, schedule = covariance_store
    formation = schedule.iloc[0]["formation_timestamp"]
    first = select_basket_covariance(
        store,
        SYNTHETIC_BASKET_ID,
        formation,
        source=SOURCE,
    )
    second = select_basket_covariance(
        store,
        SYNTHETIC_BASKET_ID,
        formation,
        source=SOURCE,
    )

    assert first == second
    assert first.downstream_eligible and first.no_lookahead_passed
    assert first.selected_estimator == "ewma"
    assert first.alternative_promotion_passed
    assert len(first.candidates) == 3
    assert len(first.folds) == 30
    assert all(candidate.stability_passed for candidate in first.candidates)
    assert all(
        pd.Timestamp(fold.training_end) < pd.Timestamp(fold.evaluation_timestamp)
        for fold in first.folds
    )
    assert pd.Timestamp(first.evaluation_end) <= pd.Timestamp(first.formation_timestamp)
    assert verify_basket_covariance_selection(first)
    assert verify_basket_covariance_selection(first.canonical_json)
    assert b'"selected_estimator": "ewma"' in basket_covariance_selection_report_json(first)
    tampered = json.loads(first.canonical_json)
    tampered["selected_estimator"] = "sample_shrinkage"
    assert not verify_basket_covariance_selection(tampered)


def test_improvement_gate_retains_baseline_without_blocking_valid_inputs(
    covariance_store: tuple[BasketInputStore, pd.DataFrame],
) -> None:
    store, schedule = covariance_store
    selection = select_basket_covariance(
        store,
        SYNTHETIC_BASKET_ID,
        schedule.iloc[0]["formation_timestamp"],
        source=SOURCE,
        settings=BasketCovarianceSelectionSettings(minimum_nll_improvement=1_000.0),
    )

    assert selection.best_scoring_estimator == "ewma"
    assert selection.selected_estimator == "sample_shrinkage"
    assert not selection.alternative_promotion_passed
    assert selection.nll_improvement_over_baseline == 0.0
    assert selection.downstream_eligible


def test_student_t_regime_validation_is_chronological_robust_and_audited(
    covariance_store: tuple[BasketInputStore, pd.DataFrame],
) -> None:
    store, schedule = covariance_store
    formation = schedule.iloc[0]["formation_timestamp"]
    settings = BasketCovarianceSelectionSettings(
        forecast_distribution="student_t",
        selection_objective="regime_balanced",
    )
    selection = select_basket_covariance(
        store,
        SYNTHETIC_BASKET_ID,
        formation,
        source=SOURCE,
        settings=settings,
    )

    assert selection.downstream_eligible
    assert selection.regime_coverage_passed and selection.robust_validation_passed
    assert selection.stress_evaluation_count == 4
    assert selection.selected_estimator == "sample_shrinkage"
    assert selection.selected_stress_resilience_passed
    assert verify_basket_covariance_selection(selection)
    assert all(
        np.isfinite(fold.gaussian_negative_log_likelihood)
        and np.isfinite(fold.student_t_negative_log_likelihood)
        and fold.forecast_distribution == "student_t"
        for fold in selection.folds
    )
    for sequence in range(settings.evaluation_returns):
        matching = [fold for fold in selection.folds if fold.sequence == sequence]
        assert len({fold.regime for fold in matching}) == 1
        assert len({fold.realized_return_energy for fold in matching}) == 1
        assert len({fold.stress_threshold for fold in matching}) == 1
        assert all(
            pd.Timestamp(fold.training_end) < pd.Timestamp(fold.evaluation_timestamp)
            for fold in matching
        )

    defensive = select_basket_covariance(
        store,
        SYNTHETIC_BASKET_ID,
        formation,
        source=SOURCE,
        settings=BasketCovarianceSelectionSettings(
            forecast_distribution="student_t",
            selection_objective="regime_balanced",
            stress_weight=0.0,
        ),
    )
    assert defensive.best_scoring_estimator == "diagonal_shrinkage"
    assert not defensive.best_scoring_stress_resilience_passed
    assert defensive.selected_estimator == "sample_shrinkage"
    assert defensive.selected_stress_resilience_passed

    insufficient = select_basket_covariance(
        store,
        SYNTHETIC_BASKET_ID,
        formation,
        source=SOURCE,
        settings=BasketCovarianceSelectionSettings(
            forecast_distribution="student_t",
            selection_objective="regime_balanced",
            minimum_stress_evaluations=9,
        ),
    )
    assert not insufficient.downstream_eligible
    assert not insufficient.regime_coverage_passed
    assert "sufficient calm and stress regimes" in insufficient.failure_reasons[0]


def test_all_estimators_create_distinct_psd_markets(
    covariance_store: tuple[BasketInputStore, pd.DataFrame],
) -> None:
    store, schedule = covariance_store
    formation = schedule.iloc[0]["formation_timestamp"]
    estimates = [
        estimate_basket_market(
            store,
            SYNTHETIC_BASKET_ID,
            formation,
            source=SOURCE,
            settings=BasketEstimationSettings(estimator=estimator),
        )
        for estimator in COVARIANCE_ESTIMATORS
    ]

    assert {estimate.settings.estimator for estimate in estimates} == set(COVARIANCE_ESTIMATORS)
    assert all(
        np.linalg.eigvalsh(np.asarray(estimate.market.correlation)).min() >= -1e-12
        for estimate in estimates
    )
    assert len({estimate.market.sha256 for estimate in estimates}) == 3


def test_selected_materialization_binds_and_is_required_by_advanced_replay(
    covariance_store: tuple[BasketInputStore, pd.DataFrame],
) -> None:
    store, schedule = covariance_store
    frame = materialize_selected_covariance_replay(
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
    settings = BasketRiskHedgeSettings(
        paths=2_000,
        maximum_absolute_option_position=150.0,
        require_covariance_selection=True,
    )
    result = run_basket_risk_hedge_replay(
        frame,
        terms,
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=settings,
    )

    assert result.downstream_eligible
    assert result.estimate_provenance_bound
    assert result.covariance_selection_provenance_bound
    assert all(snapshot.input_covariance_selection_sha256 for snapshot in result.snapshots)

    unselected = frame.drop(
        columns=[
            "input_covariance_selection_sha256",
            "input_covariance_selection_json",
        ]
    )
    with pytest.raises(BasketRiskHedgeSchemaError, match="selection provenance is required"):
        run_basket_risk_hedge_replay(
            unselected,
            terms,
            option_type="call",
            strike=60,
            expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
            settings=settings,
        )

    tampered = frame.copy()
    payload = json.loads(str(tampered.loc[0, "input_covariance_selection_json"]))
    payload["selected_estimator"] = "diagonal_shrinkage"
    tampered.loc[0, "input_covariance_selection_json"] = json.dumps(payload)
    blocked = run_basket_risk_hedge_replay(
        tampered,
        terms,
        option_type="call",
        strike=60,
        expiry_timestamp=SYNTHETIC_BASKET_EXPIRY,
        settings=settings,
    )
    assert not blocked.downstream_eligible
    assert "self-hash" in blocked.failure_reasons[0]


def test_selection_rejects_short_history_and_invalid_settings(tmp_path: Path) -> None:
    records, schedule = generate_synthetic_statistical_basket_inputs()
    recent = records.tail(15).copy()
    store = BasketInputStore.create(tmp_path / "short.sqlite")
    store.ingest_frame(
        recent,
        source_file_name="short.csv",
        source=SOURCE,
        mode="initial",
        ingested_timestamp="2026-05-01T00:00:00Z",
    )
    with pytest.raises(BasketInputIntegrityError, match="requires at least 30"):
        select_basket_covariance(
            store,
            SYNTHETIC_BASKET_ID,
            schedule.iloc[-1]["formation_timestamp"],
            source=SOURCE,
        )
    with pytest.raises(ValueError, match="training_window"):
        BasketCovarianceSelectionSettings(training_window=5, minimum_training_returns=10)
    with pytest.raises(ValueError, match="minimum_nll_improvement"):
        BasketCovarianceSelectionSettings(minimum_nll_improvement=-1.0)
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        BasketCovarianceSelectionSettings(student_t_degrees_of_freedom=2.0)
    with pytest.raises(ValueError, match="minimum_stress_evaluations"):
        BasketCovarianceSelectionSettings(minimum_stress_evaluations=11)
    assert BasketCovarianceSelectionSettings(evaluation_returns=2).selection_objective == "mean_nll"
    with pytest.raises(ValueError, match="regime_balanced"):
        BasketCovarianceSelectionSettings(
            evaluation_returns=2,
            selection_objective="regime_balanced",
        )
    with pytest.raises(ValueError, match="estimator"):
        BasketEstimationSettings(estimator="unknown")
    with pytest.raises(ValueError, match="covariance selection requires"):
        BasketRiskHedgeSettings(
            require_estimate_provenance=False,
            require_covariance_selection=True,
        )
