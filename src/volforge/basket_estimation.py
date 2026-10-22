"""Point-in-time statistical basket-market estimation from the append-only store."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from volforge.basket import BasketAssetMarket, BasketMarket
from volforge.basket_fixtures import (
    synthetic_hedge_instruments_json,
    synthetic_smile_hedge_instruments_json,
)
from volforge.basket_inputs.store import (
    BASKET_INPUT_COLUMNS,
    SYNTHETIC_BASKET_ID,
    BasketInputIntegrityError,
    BasketInputSchemaError,
    BasketInputStore,
)

COVARIANCE_ESTIMATORS = ("sample_shrinkage", "ewma", "diagonal_shrinkage")


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
class BasketEstimationSettings:
    """Rolling-window and shrinkage controls for point-in-time estimates."""

    window_returns: int = 20
    minimum_returns: int = 10
    annualization_factor: float = 252.0
    correlation_shrinkage: float = 0.20
    estimator: str = "sample_shrinkage"
    ewma_decay: float = 0.94
    covariance_shrinkage: float = 0.35
    volatility_floor: float = 1e-4

    def __post_init__(self) -> None:
        if isinstance(self.window_returns, bool) or not isinstance(self.window_returns, int):
            raise ValueError("window_returns must be an integer")
        if isinstance(self.minimum_returns, bool) or not isinstance(self.minimum_returns, int):
            raise ValueError("minimum_returns must be an integer")
        if self.minimum_returns < 2 or self.window_returns < self.minimum_returns:
            raise ValueError("window_returns must be at least minimum_returns >= 2")
        if not math.isfinite(self.annualization_factor) or self.annualization_factor <= 0.0:
            raise ValueError("annualization_factor must be positive and finite")
        if (
            not math.isfinite(self.correlation_shrinkage)
            or not 0.0 <= self.correlation_shrinkage <= 1.0
        ):
            raise ValueError("correlation_shrinkage must lie in [0, 1]")
        if self.estimator not in COVARIANCE_ESTIMATORS:
            raise ValueError("estimator must be one of " + ", ".join(COVARIANCE_ESTIMATORS))
        if not math.isfinite(self.ewma_decay) or not 0.0 < self.ewma_decay < 1.0:
            raise ValueError("ewma_decay must lie strictly between zero and one")
        if (
            not math.isfinite(self.covariance_shrinkage)
            or not 0.0 <= self.covariance_shrinkage <= 1.0
        ):
            raise ValueError("covariance_shrinkage must lie in [0, 1]")
        if not math.isfinite(self.volatility_floor) or self.volatility_floor <= 0.0:
            raise ValueError("volatility_floor must be positive and finite")


@dataclass(frozen=True, slots=True)
class BasketInputEstimate:
    """A provenance-bound basket market estimated as known at one formation time."""

    settings: BasketEstimationSettings
    basket_id: str
    source: str
    formation_timestamp: str
    dataset_known_at: str
    observation_start: str
    observation_end: str
    price_observation_count: int
    return_observation_count: int
    asset_ids: tuple[str, ...]
    annualized_volatilities: tuple[float, ...]
    sample_correlation: tuple[tuple[float, ...], ...]
    shrunk_correlation: tuple[tuple[float, ...], ...]
    sample_minimum_eigenvalue: float
    shrunk_minimum_eigenvalue: float
    no_lookahead_passed: bool
    market: BasketMarket
    execution_json: str
    input_store_snapshot_sha256: str
    input_manifest_hashes: tuple[str, ...]
    input_record_payload_sha256s: tuple[str, ...]
    estimate_sha256: str

    def _payload(self, *, include_hash: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "settings": asdict(self.settings),
            "basket_id": self.basket_id,
            "source": self.source,
            "formation_timestamp": self.formation_timestamp,
            "dataset_known_at": self.dataset_known_at,
            "observation_start": self.observation_start,
            "observation_end": self.observation_end,
            "price_observation_count": self.price_observation_count,
            "return_observation_count": self.return_observation_count,
            "asset_ids": list(self.asset_ids),
            "annualized_volatilities": list(self.annualized_volatilities),
            "sample_correlation": [list(row) for row in self.sample_correlation],
            "shrunk_correlation": [list(row) for row in self.shrunk_correlation],
            "sample_minimum_eigenvalue": self.sample_minimum_eigenvalue,
            "shrunk_minimum_eigenvalue": self.shrunk_minimum_eigenvalue,
            "no_lookahead_passed": self.no_lookahead_passed,
            "market_json": self.market.canonical_json,
            "market_sha256": self.market.sha256,
            "execution_json": self.execution_json,
            "input_store_snapshot_sha256": self.input_store_snapshot_sha256,
            "input_manifest_hashes": list(self.input_manifest_hashes),
            "input_record_payload_sha256s": list(self.input_record_payload_sha256s),
        }
        if include_hash:
            payload["estimate_sha256"] = self.estimate_sha256
        return payload

    @property
    def canonical_json(self) -> str:
        """Return the complete deterministic estimate audit as canonical JSON."""
        return _canonical_json(self._payload(include_hash=True))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete deterministic estimate audit."""
        return self._payload(include_hash=True)


def verify_basket_input_estimate(value: BasketInputEstimate | str | dict[str, Any]) -> bool:
    """Verify an estimate's self-hash without trusting its embedded identity."""
    if isinstance(value, BasketInputEstimate):
        return value.estimate_sha256 == _sha256(value._payload(include_hash=False))
    try:
        payload = json.loads(value) if isinstance(value, str) else dict(value)
        embedded_hash = payload.pop("estimate_sha256")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(embedded_hash, str) and embedded_hash == _sha256(payload)


def _complete_price_history(
    records: pd.DataFrame,
    *,
    asset_ids: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    components = records.loc[records["record_type"] == "component"].copy()
    if components.empty:
        raise BasketInputIntegrityError("estimation history contains no component observations")
    rows: list[dict[str, Any]] = []
    provenance_rows: list[pd.DataFrame] = []
    for market_time, time_group in components.groupby("market_timestamp", sort=True):
        snapshot_ids = sorted(str(value) for value in time_group["snapshot_id"].unique())
        complete: list[pd.DataFrame] = []
        for snapshot_id in snapshot_ids:
            package = time_group.loc[time_group["snapshot_id"] == snapshot_id]
            ids = tuple(sorted(str(value) for value in package["asset_id"]))
            if ids == asset_ids and not package["asset_id"].duplicated().any():
                complete.append(package)
        if len(complete) > 1:
            raise BasketInputIntegrityError(
                f"multiple complete snapshots exist at estimation time {market_time}"
            )
        if not complete:
            continue
        package = complete[0].sort_values("asset_id")
        row: dict[str, Any] = {"market_timestamp": pd.Timestamp(market_time)}
        for _, component in package.iterrows():
            spot = float(component["spot"])
            if not math.isfinite(spot) or spot <= 0.0:
                raise BasketInputIntegrityError("estimation history contains an invalid spot")
            row[str(component["asset_id"])] = spot
        rows.append(row)
        provenance_rows.append(package)
    if not rows:
        raise BasketInputIntegrityError("estimation history has no complete synchronized snapshots")
    return pd.DataFrame(rows).sort_values("market_timestamp"), pd.concat(provenance_rows)


def _correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if np.any(standard_deviations <= 0.0):
        raise BasketInputIntegrityError("estimated covariance contains zero variance")
    correlation = covariance / np.outer(standard_deviations, standard_deviations)
    correlation = np.clip(0.5 * (correlation + correlation.T), -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def estimate_return_moments(
    returns: np.ndarray,
    settings: BasketEstimationSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate daily mean/covariance plus sample and selected correlations."""
    observations = np.asarray(returns, dtype=float)
    if observations.ndim != 2 or len(observations) < settings.minimum_returns:
        raise BasketInputIntegrityError(
            f"estimation requires at least {settings.minimum_returns} synchronized returns"
        )
    if not np.isfinite(observations).all():
        raise BasketInputIntegrityError("return history contains non-finite values")
    sample_mean = np.mean(observations, axis=0)
    sample_covariance = np.atleast_2d(np.cov(observations, rowvar=False, ddof=1)).astype(float)
    sample_correlation = _correlation_from_covariance(sample_covariance)

    if settings.estimator == "sample_shrinkage":
        mean = sample_mean
        identity = np.eye(sample_covariance.shape[0], dtype=float)
        selected_correlation = (
            1.0 - settings.correlation_shrinkage
        ) * sample_correlation + settings.correlation_shrinkage * identity
        standard_deviations = np.sqrt(np.diag(sample_covariance))
        covariance = selected_correlation * np.outer(standard_deviations, standard_deviations)
    elif settings.estimator == "diagonal_shrinkage":
        mean = sample_mean
        target = np.diag(np.diag(sample_covariance))
        covariance = (
            1.0 - settings.covariance_shrinkage
        ) * sample_covariance + settings.covariance_shrinkage * target
        selected_correlation = _correlation_from_covariance(covariance)
    else:
        powers = np.arange(len(observations) - 1, -1, -1, dtype=float)
        weights = settings.ewma_decay**powers
        weights /= weights.sum()
        mean = np.sum(observations * weights[:, None], axis=0)
        centered = observations - mean
        denominator = 1.0 - float(np.sum(weights**2))
        if denominator <= 0.0:
            raise BasketInputIntegrityError("EWMA covariance has no effective degrees of freedom")
        covariance = (centered * weights[:, None]).T @ centered / denominator
        selected_correlation = _correlation_from_covariance(covariance)

    covariance = 0.5 * (covariance + covariance.T)
    selected_correlation = 0.5 * (selected_correlation + selected_correlation.T)
    np.fill_diagonal(selected_correlation, 1.0)
    if not np.isfinite(covariance).all() or not np.isfinite(selected_correlation).all():
        raise BasketInputIntegrityError("estimated covariance is not finite")
    return mean, covariance, sample_correlation, selected_correlation


def estimate_basket_market(
    store: BasketInputStore,
    basket_id: str,
    formation_timestamp: str | pd.Timestamp,
    *,
    source: str | None = None,
    maximum_age: str | pd.Timedelta = "15min",
    dataset_known_at: str | pd.Timestamp | None = None,
    settings: BasketEstimationSettings | None = None,
) -> BasketInputEstimate:
    """Estimate annualized volatilities and a shrunk correlation matrix as known."""
    settings = settings or BasketEstimationSettings()
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
    available_returns = len(history) - 1
    if available_returns < settings.minimum_returns:
        raise BasketInputIntegrityError(
            f"estimation requires at least {settings.minimum_returns} synchronized returns; "
            f"found {max(available_returns, 0)}"
        )
    history = history.tail(settings.window_returns + 1).reset_index(drop=True)
    selected_times = set(history["market_timestamp"].tolist())
    provenance = provenance.loc[provenance["market_timestamp"].isin(selected_times)].copy()
    levels = history.loc[:, list(current.market.asset_ids)].to_numpy(dtype=float)
    returns = np.diff(np.log(levels), axis=0)
    _, covariance, sample, shrunk = estimate_return_moments(returns, settings)
    annualized = np.sqrt(np.maximum(np.diag(covariance), 0.0)) * math.sqrt(
        settings.annualization_factor
    )
    annualized = np.maximum(annualized, settings.volatility_floor)
    if sample.shape != (len(current.market.assets), len(current.market.assets)):
        raise BasketInputIntegrityError("sample correlation has the wrong dimension")
    sample_min = float(np.linalg.eigvalsh(sample).min())
    shrunk_min = float(np.linalg.eigvalsh(shrunk).min())
    if shrunk_min < -1e-10:
        raise BasketInputIntegrityError("shrunk correlation matrix is not positive semidefinite")

    market = BasketMarket(
        assets=tuple(
            BasketAssetMarket(
                asset_id=asset.asset_id,
                spot=asset.spot,
                volatility=float(annualized[position]),
                dividend_yield=asset.dividend_yield,
            )
            for position, asset in enumerate(current.market.assets)
        ),
        correlation=tuple(tuple(float(item) for item in row) for row in shrunk),
    )
    max_market = pd.to_datetime(provenance["market_timestamp"], utc=True).max()
    max_received = pd.to_datetime(provenance["received_timestamp"], utc=True).max()
    no_lookahead = bool(max_market <= formation and max_received <= formation)
    if not no_lookahead:
        raise BasketInputIntegrityError("estimation history violates the no-lookahead gate")
    manifest_rows = provenance[["batch_sequence", "manifest_hash"]].drop_duplicates()
    manifest_rows = manifest_rows.sort_values("batch_sequence")
    manifest_hashes = tuple(
        sorted(
            {
                *(str(value) for value in manifest_rows["manifest_hash"]),
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
    estimate = BasketInputEstimate(
        settings=settings,
        basket_id=basket_id,
        source=current.source,
        formation_timestamp=formation.isoformat(),
        dataset_known_at=dataset_time.isoformat(),
        observation_start=pd.Timestamp(history.iloc[0]["market_timestamp"]).isoformat(),
        observation_end=pd.Timestamp(history.iloc[-1]["market_timestamp"]).isoformat(),
        price_observation_count=len(history),
        return_observation_count=len(returns),
        asset_ids=current.market.asset_ids,
        annualized_volatilities=tuple(float(value) for value in annualized),
        sample_correlation=tuple(tuple(float(item) for item in row) for row in sample),
        shrunk_correlation=tuple(tuple(float(item) for item in row) for row in shrunk),
        sample_minimum_eigenvalue=sample_min,
        shrunk_minimum_eigenvalue=shrunk_min,
        no_lookahead_passed=True,
        market=market,
        execution_json=current.execution_json,
        input_store_snapshot_sha256=current.snapshot_sha256,
        input_manifest_hashes=manifest_hashes,
        input_record_payload_sha256s=record_hashes,
        estimate_sha256="",
    )
    return replace(estimate, estimate_sha256=_sha256(estimate._payload(include_hash=False)))


def materialize_estimated_replay(
    store: BasketInputStore,
    schedule: pd.DataFrame,
    *,
    basket_id: str,
    source: str | None = None,
    maximum_age: str | pd.Timedelta = "15min",
    dataset_known_at: str | pd.Timestamp | None = None,
    settings: BasketEstimationSettings | None = None,
) -> pd.DataFrame:
    """Resolve a replay schedule with statistically estimated, provenance-bound markets."""
    from volforge.basket_replay import BASKET_REPLAY_COLUMNS

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
        estimate = estimate_basket_market(
            store,
            basket_id,
            row["formation_timestamp"],
            source=source,
            maximum_age=maximum_age,
            dataset_known_at=dataset_known_at,
            settings=settings,
        )
        current = store.resolve_snapshot(
            basket_id,
            row["formation_timestamp"],
            source=estimate.source,
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
            *passthrough,
        ),
    )


def _generate_synthetic_statistical_basket_inputs(
    *,
    include_smile_hedges: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from volforge.basket_replay import generate_synthetic_basket_replay

    replay = generate_synthetic_basket_replay()
    anchor_spots = {
        pd.Timestamp(row["formation_timestamp"]).normalize(): tuple(
            asset["spot"] for asset in json.loads(str(row["market_json"]))["assets"]
        )
        for _, row in replay.iterrows()
    }
    dates = pd.bdate_range("2026-02-16", "2026-04-30", tz="UTC")
    rng = np.random.default_rng(1701)
    innovations = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=np.asarray([[1.0, 0.35], [0.35, 1.0]]),
        size=len(dates),
    )
    levels = np.empty((len(dates), 2), dtype=float)
    levels[0] = (112.0, 18.0)
    daily_vol = np.asarray([0.24, 0.34]) / math.sqrt(252.0)
    for index in range(1, len(dates)):
        levels[index] = levels[index - 1] * np.exp(
            -0.5 * daily_vol**2 + daily_vol * innovations[index]
        )
    for index, date in enumerate(dates):
        if date.normalize() in anchor_spots:
            levels[index] = anchor_spots[date.normalize()]

    records: list[dict[str, Any]] = []
    for index, (date, spot_pair) in enumerate(zip(dates, levels, strict=True)):
        market_time = date + pd.Timedelta(hours=19, minutes=58)
        received = market_time + pd.Timedelta(minutes=1)
        snapshot_id = f"SYN-STAT-{index + 1:04d}"
        for asset_id, spot, volatility, dividend in (
            ("ACME", spot_pair[0], 0.24, 0.01),
            ("SPIN", spot_pair[1], 0.34, 0.0),
        ):
            records.append(
                {
                    "record_type": "component",
                    "snapshot_id": snapshot_id,
                    "market_timestamp": market_time.isoformat(),
                    "received_timestamp": received.isoformat(),
                    "source_record_id": f"{snapshot_id}:component:{asset_id}",
                    "revision": 0,
                    "record_action": "upsert",
                    "basket_id": SYNTHETIC_BASKET_ID,
                    "asset_id": asset_id,
                    "paired_asset_id": None,
                    "spot": float(spot),
                    "volatility": volatility,
                    "dividend_yield": dividend,
                    "bid": float(spot) - 0.02,
                    "ask": float(spot) + 0.02,
                    "correlation": None,
                }
            )
        records.append(
            {
                "record_type": "correlation",
                "snapshot_id": snapshot_id,
                "market_timestamp": market_time.isoformat(),
                "received_timestamp": received.isoformat(),
                "source_record_id": f"{snapshot_id}:correlation:ACME:SPIN",
                "revision": 0,
                "record_action": "upsert",
                "basket_id": SYNTHETIC_BASKET_ID,
                "asset_id": "ACME",
                "paired_asset_id": "SPIN",
                "spot": None,
                "volatility": None,
                "dividend_yield": None,
                "bid": None,
                "ask": None,
                "correlation": 0.35,
            }
        )
    schedule = replay[["formation_timestamp", "option_bid", "option_ask", "risk_free_rate"]].copy()
    instrument_generator = (
        synthetic_smile_hedge_instruments_json
        if include_smile_hedges
        else synthetic_hedge_instruments_json
    )
    schedule["hedge_instruments_json"] = [
        instrument_generator(
            row["market_json"],
            formation_timestamp=row["formation_timestamp"],
            rate=float(row["risk_free_rate"]),
        )
        for _, row in replay.iterrows()
    ]
    return pd.DataFrame(records, columns=BASKET_INPUT_COLUMNS), schedule


def generate_synthetic_statistical_basket_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate daily history plus a three-risk advanced hedge schedule."""
    return _generate_synthetic_statistical_basket_inputs(include_smile_hedges=False)


def generate_synthetic_smile_basket_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate daily history plus a five-risk dynamic-smile hedge schedule."""
    return _generate_synthetic_statistical_basket_inputs(include_smile_hedges=True)


__all__ = [
    "COVARIANCE_ESTIMATORS",
    "BasketEstimationSettings",
    "BasketInputEstimate",
    "estimate_basket_market",
    "estimate_return_moments",
    "generate_synthetic_smile_basket_inputs",
    "generate_synthetic_statistical_basket_inputs",
    "materialize_estimated_replay",
    "verify_basket_input_estimate",
]
