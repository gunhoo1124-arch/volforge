"""Chronological out-of-sample covariance selection for basket markets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from volforge.basket_estimation import (
    COVARIANCE_ESTIMATORS,
    BasketEstimationSettings,
    _complete_price_history,
    estimate_basket_market,
    estimate_return_moments,
)
from volforge.basket_inputs.store import (
    BasketInputIntegrityError,
    BasketInputSchemaError,
    BasketInputStore,
)

COVARIANCE_SELECTION_COLUMNS = (
    "input_covariance_selection_sha256",
    "input_covariance_selection_json",
)
FORECAST_DISTRIBUTIONS = ("gaussian", "student_t")
SELECTION_OBJECTIVES = ("mean_nll", "regime_balanced")


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


def _timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise BasketInputSchemaError(f"{name} must be a timezone-aware timestamp") from exc
    if pd.isna(result) or result.tzinfo is None:
        raise BasketInputSchemaError(f"{name} must be a timezone-aware timestamp")
    return result.tz_convert("UTC")


@dataclass(frozen=True, slots=True)
class BasketCovarianceSelectionSettings:
    """Rolling forecast design, candidate parameters, and promotion gates."""

    training_window: int = 20
    minimum_training_returns: int = 10
    evaluation_returns: int = 10
    annualization_factor: float = 252.0
    correlation_shrinkage: float = 0.20
    covariance_shrinkage: float = 0.35
    ewma_decay: float = 0.94
    volatility_floor: float = 1e-4
    eigenvalue_floor: float = 1e-10
    maximum_condition_number: float = 1e8
    minimum_nll_improvement: float = 0.0
    forecast_distribution: str = "gaussian"
    student_t_degrees_of_freedom: float = 6.0
    selection_objective: str = "mean_nll"
    stress_quantile: float = 0.80
    stress_weight: float = 0.50
    minimum_stress_evaluations: int = 2
    maximum_stress_nll_regret: float = 0.0

    def __post_init__(self) -> None:
        integers = {
            "training_window": self.training_window,
            "minimum_training_returns": self.minimum_training_returns,
            "evaluation_returns": self.evaluation_returns,
            "minimum_stress_evaluations": self.minimum_stress_evaluations,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in integers.values()
        ):
            raise ValueError("covariance selection windows must be integers")
        if self.minimum_training_returns < 2:
            raise ValueError("minimum_training_returns must be at least two")
        if self.training_window < self.minimum_training_returns:
            raise ValueError("training_window must be at least minimum_training_returns")
        if self.evaluation_returns < 2:
            raise ValueError("evaluation_returns must be at least two")
        if not 1 <= self.minimum_stress_evaluations <= self.evaluation_returns:
            raise ValueError(
                "minimum_stress_evaluations must be positive and at most evaluation_returns"
            )
        positive = {
            "annualization_factor": self.annualization_factor,
            "volatility_floor": self.volatility_floor,
            "eigenvalue_floor": self.eigenvalue_floor,
            "maximum_condition_number": self.maximum_condition_number,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        unit_interval = {
            "correlation_shrinkage": self.correlation_shrinkage,
            "covariance_shrinkage": self.covariance_shrinkage,
        }
        for name, value in unit_interval.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not math.isfinite(self.ewma_decay) or not 0.0 < self.ewma_decay < 1.0:
            raise ValueError("ewma_decay must lie strictly between zero and one")
        if not math.isfinite(self.minimum_nll_improvement) or self.minimum_nll_improvement < 0.0:
            raise ValueError("minimum_nll_improvement must be non-negative and finite")
        if self.forecast_distribution not in FORECAST_DISTRIBUTIONS:
            raise ValueError(
                "forecast_distribution must be one of " + ", ".join(FORECAST_DISTRIBUTIONS)
            )
        if (
            not math.isfinite(self.student_t_degrees_of_freedom)
            or self.student_t_degrees_of_freedom <= 2.0
            or self.student_t_degrees_of_freedom > 1e6
        ):
            raise ValueError("student_t_degrees_of_freedom must lie in (2, 1e6]")
        if self.selection_objective not in SELECTION_OBJECTIVES:
            raise ValueError(
                "selection_objective must be one of " + ", ".join(SELECTION_OBJECTIVES)
            )
        if (
            self.selection_objective == "regime_balanced"
            and self.minimum_stress_evaluations >= self.evaluation_returns
        ):
            raise ValueError(
                "regime_balanced selection requires minimum_stress_evaluations below "
                "evaluation_returns"
            )
        if not math.isfinite(self.stress_quantile) or not 0.0 < self.stress_quantile < 1.0:
            raise ValueError("stress_quantile must lie strictly between zero and one")
        if not math.isfinite(self.stress_weight) or not 0.0 <= self.stress_weight <= 1.0:
            raise ValueError("stress_weight must lie in [0, 1]")
        if (
            not math.isfinite(self.maximum_stress_nll_regret)
            or self.maximum_stress_nll_regret < 0.0
        ):
            raise ValueError("maximum_stress_nll_regret must be non-negative and finite")

    def estimator_settings(self, estimator: str) -> BasketEstimationSettings:
        """Map a selected candidate to the standard audited estimator settings."""
        return BasketEstimationSettings(
            window_returns=self.training_window,
            minimum_returns=self.minimum_training_returns,
            annualization_factor=self.annualization_factor,
            correlation_shrinkage=self.correlation_shrinkage,
            estimator=estimator,
            ewma_decay=self.ewma_decay,
            covariance_shrinkage=self.covariance_shrinkage,
            volatility_floor=self.volatility_floor,
        )


@dataclass(frozen=True, slots=True)
class CovarianceForecastFold:
    """One strictly forward one-return covariance forecast score."""

    estimator: str
    sequence: int
    training_start: str
    training_end: str
    evaluation_timestamp: str
    training_return_count: int
    forecast_distribution: str
    regime: str
    realized_return_energy: float
    stress_threshold: float
    gaussian_negative_log_likelihood: float
    student_t_negative_log_likelihood: float
    negative_log_likelihood: float
    raw_minimum_eigenvalue: float
    stabilized_minimum_eigenvalue: float
    condition_number: float
    stability_passed: bool


@dataclass(frozen=True, slots=True)
class CovarianceCandidateSummary:
    """Aggregated out-of-sample loss and numerical diagnostics for one candidate."""

    estimator: str
    evaluation_count: int
    calm_evaluation_count: int
    stress_evaluation_count: int
    mean_negative_log_likelihood: float
    median_negative_log_likelihood: float
    calm_mean_negative_log_likelihood: float | None
    stress_mean_negative_log_likelihood: float | None
    worst_negative_log_likelihood: float
    selection_score: float
    minimum_raw_eigenvalue: float
    maximum_condition_number: float
    formation_raw_minimum_eigenvalue: float
    formation_stabilized_minimum_eigenvalue: float
    formation_condition_number: float
    formation_stability_passed: bool
    stability_passed: bool


@dataclass(frozen=True, slots=True)
class BasketCovarianceSelection:
    """Tamper-evident point-in-time covariance model selection audit."""

    settings: BasketCovarianceSelectionSettings
    basket_id: str
    source: str
    formation_timestamp: str
    dataset_known_at: str
    observation_start: str
    observation_end: str
    evaluation_start: str
    evaluation_end: str
    asset_ids: tuple[str, ...]
    candidates: tuple[CovarianceCandidateSummary, ...]
    folds: tuple[CovarianceForecastFold, ...]
    baseline_estimator: str
    best_scoring_estimator: str
    selected_estimator: str
    baseline_mean_negative_log_likelihood: float
    selected_mean_negative_log_likelihood: float
    nll_improvement_over_baseline: float
    baseline_selection_score: float
    selected_selection_score: float
    selection_score_improvement: float
    stress_evaluation_count: int
    regime_coverage_passed: bool
    best_scoring_stress_nll_regret: float | None
    best_scoring_stress_resilience_passed: bool
    selected_stress_nll_regret: float | None
    selected_stress_resilience_passed: bool
    robust_validation_passed: bool
    alternative_promotion_passed: bool
    selected_stability_passed: bool
    no_lookahead_passed: bool
    downstream_eligible: bool
    failure_reasons: tuple[str, ...]
    input_store_snapshot_sha256: str
    input_manifest_hashes: tuple[str, ...]
    input_record_payload_sha256s: tuple[str, ...]
    selection_sha256: str

    def _payload(self, *, include_hash: bool) -> dict[str, Any]:
        payload = asdict(self)
        payload["settings"] = asdict(self.settings)
        payload["asset_ids"] = list(self.asset_ids)
        payload["candidates"] = [asdict(item) for item in self.candidates]
        payload["folds"] = [asdict(item) for item in self.folds]
        payload["failure_reasons"] = list(self.failure_reasons)
        payload["input_manifest_hashes"] = list(self.input_manifest_hashes)
        payload["input_record_payload_sha256s"] = list(self.input_record_payload_sha256s)
        if not include_hash:
            payload.pop("selection_sha256")
        return payload

    @property
    def canonical_json(self) -> str:
        """Return the complete selection audit as canonical JSON."""
        return _canonical_json(self._payload(include_hash=True))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible audit."""
        return self._payload(include_hash=True)

    def candidate_frame(self) -> pd.DataFrame:
        """Return one comparison row per candidate."""
        return pd.DataFrame([asdict(item) for item in self.candidates])

    def fold_frame(self) -> pd.DataFrame:
        """Return one row per chronological candidate forecast."""
        return pd.DataFrame([asdict(item) for item in self.folds])


def verify_basket_covariance_selection(
    value: BasketCovarianceSelection | str | dict[str, Any],
) -> bool:
    """Verify a covariance-selection audit's canonical self-hash."""
    if isinstance(value, BasketCovarianceSelection):
        return value.selection_sha256 == _sha256(value._payload(include_hash=False))
    try:
        payload = json.loads(value) if isinstance(value, str) else dict(value)
        embedded_hash = payload.pop("selection_sha256")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(embedded_hash, str) and embedded_hash == _sha256(payload)


def _stabilize_covariance(
    covariance: np.ndarray,
    *,
    eigenvalue_floor: float,
) -> tuple[np.ndarray, float, float, float]:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    raw_minimum = float(eigenvalues.min())
    clipped = np.maximum(eigenvalues, eigenvalue_floor)
    stabilized = (eigenvectors * clipped) @ eigenvectors.T
    stabilized = 0.5 * (stabilized + stabilized.T)
    stabilized_minimum = float(clipped.min())
    condition = float(clipped.max() / clipped.min())
    return stabilized, raw_minimum, stabilized_minimum, condition


def _forecast_scores(
    evaluation_return: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    student_t_degrees_of_freedom: float,
) -> tuple[float, float]:
    """Return Gaussian and covariance-consistent multivariate Student-t NLL."""
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not math.isfinite(float(log_determinant)):
        raise ArithmeticError("forecast covariance determinant is not positive")
    residual = evaluation_return - mean
    quadratic = max(float(residual @ np.linalg.solve(covariance, residual)), 0.0)
    dimensions = len(residual)
    gaussian = 0.5 * (dimensions * math.log(2.0 * math.pi) + float(log_determinant) + quadratic)
    degrees = student_t_degrees_of_freedom
    student_t = (
        math.lgamma(degrees / 2.0)
        - math.lgamma((degrees + dimensions) / 2.0)
        + 0.5 * (dimensions * math.log((degrees - 2.0) * math.pi) + float(log_determinant))
        + 0.5 * (degrees + dimensions) * math.log1p(quadratic / (degrees - 2.0))
    )
    return float(gaussian), float(student_t)


def _return_regime(
    training_returns: np.ndarray,
    evaluation_return: np.ndarray,
    *,
    stress_quantile: float,
) -> tuple[str, float, float]:
    """Classify a return using a robust energy threshold fitted only on earlier returns."""
    center = np.median(training_returns, axis=0)
    median_absolute_deviation = np.median(np.abs(training_returns - center), axis=0)
    robust_scale = 1.4826 * median_absolute_deviation
    fallback_scale = np.std(training_returns, axis=0, ddof=1)
    scale = np.where(robust_scale > 1e-12, robust_scale, fallback_scale)
    scale = np.maximum(scale, 1e-12)
    training_energy = np.sqrt(np.mean(((training_returns - center) / scale) ** 2, axis=1))
    threshold = float(np.quantile(training_energy, stress_quantile))
    realized_energy = float(np.sqrt(np.mean(((evaluation_return - center) / scale) ** 2)))
    regime = "stress" if realized_energy >= threshold else "calm"
    return regime, realized_energy, threshold


def select_basket_covariance(
    store: BasketInputStore,
    basket_id: str,
    formation_timestamp: str | pd.Timestamp,
    *,
    source: str | None = None,
    maximum_age: str | pd.Timedelta = "15min",
    dataset_known_at: str | pd.Timestamp | None = None,
    settings: BasketCovarianceSelectionSettings | None = None,
) -> BasketCovarianceSelection:
    """Select a stable covariance estimator using only later held-out past returns."""
    settings = settings or BasketCovarianceSelectionSettings()
    formation = _timestamp(formation_timestamp, "formation_timestamp")
    verification = store.verify()
    if not verification.valid:
        raise BasketInputIntegrityError(
            "basket-input store failed integrity verification: "
            + "; ".join(verification.failure_reasons)
        )
    current = store.resolve_snapshot(
        basket_id,
        formation,
        source=source,
        maximum_age=maximum_age,
        dataset_known_at=dataset_known_at,
        _store_verified=True,
    )
    records = store.records_as_known(
        formation,
        basket_id=basket_id,
        source=current.source,
        dataset_known_at=dataset_known_at,
    )
    history, provenance = _complete_price_history(records, asset_ids=current.market.asset_ids)
    required_returns = settings.training_window + settings.evaluation_returns
    available_returns = len(history) - 1
    if available_returns < required_returns:
        raise BasketInputIntegrityError(
            f"covariance selection requires at least {required_returns} synchronized returns; "
            f"found {max(available_returns, 0)}"
        )
    history = history.tail(required_returns + 1).reset_index(drop=True)
    selected_times = set(history["market_timestamp"].tolist())
    provenance = provenance.loc[provenance["market_timestamp"].isin(selected_times)].copy()
    levels = history.loc[:, list(current.market.asset_ids)].to_numpy(dtype=float)
    returns = np.diff(np.log(levels), axis=0)
    return_times = pd.to_datetime(history["market_timestamp"].iloc[1:], utc=True).reset_index(
        drop=True
    )

    folds: list[CovarianceForecastFold] = []
    summaries: list[CovarianceCandidateSummary] = []
    evaluation_offset = len(returns) - settings.evaluation_returns
    regimes: dict[int, tuple[str, float, float]] = {}
    for sequence, evaluation_index in enumerate(range(evaluation_offset, len(returns))):
        training_start = evaluation_index - settings.training_window
        regimes[sequence] = _return_regime(
            returns[training_start:evaluation_index],
            returns[evaluation_index],
            stress_quantile=settings.stress_quantile,
        )
    stress_evaluation_count = sum(regime == "stress" for regime, _, _ in regimes.values())
    calm_evaluation_count = settings.evaluation_returns - stress_evaluation_count
    regime_coverage_passed = bool(
        stress_evaluation_count >= settings.minimum_stress_evaluations
        and calm_evaluation_count >= 1
    )
    for estimator in COVARIANCE_ESTIMATORS:
        estimator_settings = settings.estimator_settings(estimator)
        candidate_folds: list[CovarianceForecastFold] = []
        for sequence, evaluation_index in enumerate(range(evaluation_offset, len(returns))):
            training_start = evaluation_index - settings.training_window
            training = returns[training_start:evaluation_index]
            mean, covariance, _, _ = estimate_return_moments(training, estimator_settings)
            stabilized, raw_minimum, stabilized_minimum, condition = _stabilize_covariance(
                covariance,
                eigenvalue_floor=settings.eigenvalue_floor,
            )
            gaussian_score, student_t_score = _forecast_scores(
                returns[evaluation_index],
                mean,
                stabilized,
                student_t_degrees_of_freedom=settings.student_t_degrees_of_freedom,
            )
            score = (
                gaussian_score if settings.forecast_distribution == "gaussian" else student_t_score
            )
            regime, realized_energy, stress_threshold = regimes[sequence]
            stable = bool(
                raw_minimum >= -settings.eigenvalue_floor
                and condition <= settings.maximum_condition_number
                and all(math.isfinite(value) for value in (gaussian_score, student_t_score, score))
            )
            fold = CovarianceForecastFold(
                estimator=estimator,
                sequence=sequence,
                training_start=return_times.iloc[training_start].isoformat(),
                training_end=return_times.iloc[evaluation_index - 1].isoformat(),
                evaluation_timestamp=return_times.iloc[evaluation_index].isoformat(),
                training_return_count=len(training),
                forecast_distribution=settings.forecast_distribution,
                regime=regime,
                realized_return_energy=realized_energy,
                stress_threshold=stress_threshold,
                gaussian_negative_log_likelihood=gaussian_score,
                student_t_negative_log_likelihood=student_t_score,
                negative_log_likelihood=float(score),
                raw_minimum_eigenvalue=raw_minimum,
                stabilized_minimum_eigenvalue=stabilized_minimum,
                condition_number=condition,
                stability_passed=stable,
            )
            candidate_folds.append(fold)
            folds.append(fold)
        scores = np.asarray([item.negative_log_likelihood for item in candidate_folds], dtype=float)
        calm_scores = np.asarray(
            [item.negative_log_likelihood for item in candidate_folds if item.regime == "calm"],
            dtype=float,
        )
        stress_scores = np.asarray(
            [item.negative_log_likelihood for item in candidate_folds if item.regime == "stress"],
            dtype=float,
        )
        calm_mean = float(np.mean(calm_scores)) if len(calm_scores) else None
        stress_mean = float(np.mean(stress_scores)) if len(stress_scores) else None
        selection_score = float(np.mean(scores))
        if (
            settings.selection_objective == "regime_balanced"
            and calm_mean is not None
            and stress_mean is not None
        ):
            selection_score = (
                1.0 - settings.stress_weight
            ) * calm_mean + settings.stress_weight * stress_mean
        _, formation_covariance, _, _ = estimate_return_moments(
            returns[-settings.training_window :], estimator_settings
        )
        (
            _,
            formation_raw_minimum,
            formation_stabilized_minimum,
            formation_condition,
        ) = _stabilize_covariance(
            formation_covariance,
            eigenvalue_floor=settings.eigenvalue_floor,
        )
        formation_stable = bool(
            formation_raw_minimum >= -settings.eigenvalue_floor
            and formation_condition <= settings.maximum_condition_number
        )
        summaries.append(
            CovarianceCandidateSummary(
                estimator=estimator,
                evaluation_count=len(candidate_folds),
                calm_evaluation_count=len(calm_scores),
                stress_evaluation_count=len(stress_scores),
                mean_negative_log_likelihood=float(np.mean(scores)),
                median_negative_log_likelihood=float(np.median(scores)),
                calm_mean_negative_log_likelihood=calm_mean,
                stress_mean_negative_log_likelihood=stress_mean,
                worst_negative_log_likelihood=float(np.max(scores)),
                selection_score=float(selection_score),
                minimum_raw_eigenvalue=min(item.raw_minimum_eigenvalue for item in candidate_folds),
                maximum_condition_number=max(item.condition_number for item in candidate_folds),
                formation_raw_minimum_eigenvalue=formation_raw_minimum,
                formation_stabilized_minimum_eigenvalue=formation_stabilized_minimum,
                formation_condition_number=formation_condition,
                formation_stability_passed=formation_stable,
                stability_passed=formation_stable
                and all(item.stability_passed for item in candidate_folds),
            )
        )

    baseline_name = "sample_shrinkage"
    by_name = {item.estimator: item for item in summaries}
    baseline = by_name[baseline_name]
    stable_candidates = [item for item in summaries if item.stability_passed]
    if not stable_candidates:
        raise BasketInputIntegrityError("all covariance candidates failed stability gates")
    best = min(
        stable_candidates,
        key=lambda item: (
            item.selection_score,
            COVARIANCE_ESTIMATORS.index(item.estimator),
        ),
    )
    improvement = baseline.selection_score - best.selection_score
    best_stress_regret = (
        None
        if best.stress_mean_negative_log_likelihood is None
        or baseline.stress_mean_negative_log_likelihood is None
        else (
            best.stress_mean_negative_log_likelihood - baseline.stress_mean_negative_log_likelihood
        )
    )
    best_stress_resilience = bool(
        settings.selection_objective != "regime_balanced"
        or (
            regime_coverage_passed
            and best_stress_regret is not None
            and best_stress_regret <= settings.maximum_stress_nll_regret
        )
    )
    promotion_passed = bool(
        best.estimator != baseline_name
        and baseline.stability_passed
        and improvement >= settings.minimum_nll_improvement
        and best_stress_resilience
    )
    selected_name = best.estimator if promotion_passed else baseline_name
    selected = by_name[selected_name]
    selected_stress_regret = (
        None
        if selected.stress_mean_negative_log_likelihood is None
        or baseline.stress_mean_negative_log_likelihood is None
        else (
            selected.stress_mean_negative_log_likelihood
            - baseline.stress_mean_negative_log_likelihood
        )
    )
    selected_stress_resilience = bool(
        settings.selection_objective != "regime_balanced"
        or (
            regime_coverage_passed
            and selected_stress_regret is not None
            and selected_stress_regret <= settings.maximum_stress_nll_regret
        )
    )
    robust_validation_passed = bool(
        settings.selection_objective != "regime_balanced"
        or (regime_coverage_passed and selected_stress_resilience)
    )

    max_market = pd.to_datetime(provenance["market_timestamp"], utc=True).max()
    max_received = pd.to_datetime(provenance["received_timestamp"], utc=True).max()
    no_lookahead = bool(max_market <= formation and max_received <= formation)
    failures: list[str] = []
    if not no_lookahead:
        failures.append("selection history violates the no-lookahead gate")
    if not selected.stability_passed:
        failures.append("selected covariance estimator failed PSD or conditioning gates")
    if settings.selection_objective == "regime_balanced" and not regime_coverage_passed:
        failures.append("held-out history does not provide sufficient calm and stress regimes")
    if settings.selection_objective == "regime_balanced" and not selected_stress_resilience:
        failures.append("selected covariance estimator failed the stress-loss regret gate")
    manifest_hashes = tuple(
        sorted(
            {
                *(str(value) for value in provenance["manifest_hash"]),
                *current.manifest_hashes,
            }
        )
    )
    record_hashes = tuple(
        sorted(
            {
                *(str(value) for value in provenance["payload_sha256"]),
                *current.record_payload_sha256s,
            }
        )
    )
    dataset_time = max(
        pd.to_datetime(provenance["ingested_timestamp"], utc=True).max(),
        pd.Timestamp(current.dataset_known_at),
    )
    selection = BasketCovarianceSelection(
        settings=settings,
        basket_id=basket_id,
        source=current.source,
        formation_timestamp=formation.isoformat(),
        dataset_known_at=dataset_time.isoformat(),
        observation_start=pd.Timestamp(history.iloc[0]["market_timestamp"]).isoformat(),
        observation_end=pd.Timestamp(history.iloc[-1]["market_timestamp"]).isoformat(),
        evaluation_start=return_times.iloc[evaluation_offset].isoformat(),
        evaluation_end=return_times.iloc[-1].isoformat(),
        asset_ids=current.market.asset_ids,
        candidates=tuple(summaries),
        folds=tuple(folds),
        baseline_estimator=baseline_name,
        best_scoring_estimator=best.estimator,
        selected_estimator=selected_name,
        baseline_mean_negative_log_likelihood=baseline.mean_negative_log_likelihood,
        selected_mean_negative_log_likelihood=selected.mean_negative_log_likelihood,
        nll_improvement_over_baseline=(
            baseline.mean_negative_log_likelihood - selected.mean_negative_log_likelihood
        ),
        baseline_selection_score=baseline.selection_score,
        selected_selection_score=selected.selection_score,
        selection_score_improvement=(baseline.selection_score - selected.selection_score),
        stress_evaluation_count=stress_evaluation_count,
        regime_coverage_passed=regime_coverage_passed,
        best_scoring_stress_nll_regret=best_stress_regret,
        best_scoring_stress_resilience_passed=best_stress_resilience,
        selected_stress_nll_regret=selected_stress_regret,
        selected_stress_resilience_passed=selected_stress_resilience,
        robust_validation_passed=robust_validation_passed,
        alternative_promotion_passed=promotion_passed,
        selected_stability_passed=selected.stability_passed,
        no_lookahead_passed=no_lookahead,
        downstream_eligible=not failures,
        failure_reasons=tuple(failures),
        input_store_snapshot_sha256=current.snapshot_sha256,
        input_manifest_hashes=manifest_hashes,
        input_record_payload_sha256s=record_hashes,
        selection_sha256="",
    )
    return replace(
        selection,
        selection_sha256=_sha256(selection._payload(include_hash=False)),
    )


def materialize_selected_covariance_replay(
    store: BasketInputStore,
    schedule: pd.DataFrame,
    *,
    basket_id: str,
    source: str | None = None,
    maximum_age: str | pd.Timedelta = "15min",
    dataset_known_at: str | pd.Timestamp | None = None,
    settings: BasketCovarianceSelectionSettings | None = None,
) -> pd.DataFrame:
    """Materialize replay rows with estimate and covariance-selection provenance."""
    from volforge.basket_replay import BASKET_REPLAY_COLUMNS

    settings = settings or BasketCovarianceSelectionSettings()
    required = {"formation_timestamp", "option_bid", "option_ask", "risk_free_rate"}
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise BasketInputSchemaError(
            "missing basket replay schedule columns: " + ", ".join(missing)
        )
    passthrough = [
        column
        for column in (
            "hedge_instruments_json",
            "input_option_quote_snapshot_sha256",
            "input_option_quote_snapshot_json",
            "input_option_quote_manifest_hashes_json",
        )
        if column in schedule
    ]
    rows: list[dict[str, Any]] = []
    for _, row in schedule.iterrows():
        selection = select_basket_covariance(
            store,
            basket_id,
            row["formation_timestamp"],
            source=source,
            maximum_age=maximum_age,
            dataset_known_at=dataset_known_at,
            settings=settings,
        )
        if not selection.downstream_eligible:
            raise BasketInputIntegrityError(
                "covariance selection is not downstream eligible: "
                + "; ".join(selection.failure_reasons)
            )
        estimate = estimate_basket_market(
            store,
            basket_id,
            row["formation_timestamp"],
            source=selection.source,
            maximum_age=maximum_age,
            dataset_known_at=dataset_known_at,
            settings=settings.estimator_settings(selection.selected_estimator),
        )
        current = store.resolve_snapshot(
            basket_id,
            row["formation_timestamp"],
            source=selection.source,
            maximum_age=maximum_age,
            dataset_known_at=dataset_known_at,
        )
        materialized: dict[str, Any] = {
            "formation_timestamp": estimate.formation_timestamp,
            "market_timestamp": current.market_timestamp,
            "received_timestamp": current.received_timestamp,
            "source": estimate.source,
            "source_record_id": current.snapshot_id,
            "option_bid": float(row["option_bid"]),
            "option_ask": float(row["option_ask"]),
            "risk_free_rate": float(row["risk_free_rate"]),
            "market_json": estimate.market.canonical_json,
            "execution_json": estimate.execution_json,
            "input_store_snapshot_sha256": estimate.input_store_snapshot_sha256,
            "input_manifest_hashes_json": _canonical_json(estimate.input_manifest_hashes),
            "input_estimate_sha256": estimate.estimate_sha256,
            "input_estimate_json": estimate.canonical_json,
            "input_covariance_selection_sha256": selection.selection_sha256,
            "input_covariance_selection_json": selection.canonical_json,
        }
        for column in passthrough:
            materialized[column] = row[column]
        rows.append(materialized)
    return pd.DataFrame(
        rows,
        columns=(
            *BASKET_REPLAY_COLUMNS,
            "input_store_snapshot_sha256",
            "input_manifest_hashes_json",
            "input_estimate_sha256",
            "input_estimate_json",
            *COVARIANCE_SELECTION_COLUMNS,
            *passthrough,
        ),
    )


def basket_covariance_selection_report_json(
    selection: BasketCovarianceSelection,
) -> bytes:
    """Serialize a readable deterministic covariance-selection report."""
    return (json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")


__all__ = [
    "COVARIANCE_SELECTION_COLUMNS",
    "FORECAST_DISTRIBUTIONS",
    "SELECTION_OBJECTIVES",
    "BasketCovarianceSelection",
    "BasketCovarianceSelectionSettings",
    "CovarianceCandidateSummary",
    "CovarianceForecastFold",
    "basket_covariance_selection_report_json",
    "materialize_selected_covariance_replay",
    "select_basket_covariance",
    "verify_basket_covariance_selection",
]
