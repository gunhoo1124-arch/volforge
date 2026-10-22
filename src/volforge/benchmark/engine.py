"""Strictly chronological BSM-versus-Heston option-pricing benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import binomtest, t

from volforge.calibration import calibrate_heston
from volforge.data.chain import REQUIRED_COLUMNS, ChainSchemaError, enrich_option_chain
from volforge.models import black_scholes
from volforge.models.heston import COSSettings, HestonParameters, cos_prices
from volforge.replay import snapshot_sha256
from volforge.surfaces import calibrate_ssvi, calibrate_svi_surface, prepare_surface_quotes

COMPARISON_LOSSES = (
    "squared_error",
    "absolute_error",
    "squared_vega_scaled_error",
    "quote_band_loss",
)


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    """Training-window, calibration, pricing, and audit controls."""

    training_window: int = 2
    surface_model: str = "ssvi"
    surface_grid_points: int = 81
    heston_moneyness_min: float = -0.3
    heston_moneyness_max: float = 0.3
    heston_moneyness_points: int = 5
    heston_max_function_evaluations: int = 120
    require_feller: bool = False
    cos_terms: int = 96
    cos_truncation_width: float = 16.0
    bsm_volatility_min: float = 0.005
    bsm_volatility_max: float = 3.0
    bsm_max_function_evaluations: int = 200
    comparison_loss: str = "squared_error"
    bootstrap_resamples: int = 2000
    bootstrap_block_length: int | None = None
    bootstrap_seed: int = 7
    comparison_confidence_level: float = 0.95
    minimum_comparison_dates: int = 5
    comparison_hac_lags: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.training_window, int) or self.training_window < 1:
            raise ValueError("training_window must be a positive integer")
        if self.surface_model not in {"svi", "ssvi"}:
            raise ValueError("surface_model must be 'svi' or 'ssvi'")
        if not isinstance(self.surface_grid_points, int) or self.surface_grid_points < 21:
            raise ValueError("surface_grid_points must be an integer of at least 21")
        if (
            not isinstance(self.heston_moneyness_points, int)
            or self.heston_moneyness_points < 5
            or self.heston_moneyness_min >= self.heston_moneyness_max
        ):
            raise ValueError("Heston moneyness grid requires at least five increasing points")
        if (
            not isinstance(self.heston_max_function_evaluations, int)
            or self.heston_max_function_evaluations <= 0
        ):
            raise ValueError("heston_max_function_evaluations must be positive")
        COSSettings(self.cos_terms, self.cos_truncation_width)
        if (
            not math.isfinite(self.bsm_volatility_min)
            or not math.isfinite(self.bsm_volatility_max)
            or self.bsm_volatility_min <= 0.0
            or self.bsm_volatility_min >= self.bsm_volatility_max
        ):
            raise ValueError("BSM volatility bounds must be positive, finite, and increasing")
        if (
            not isinstance(self.bsm_max_function_evaluations, int)
            or self.bsm_max_function_evaluations <= 0
        ):
            raise ValueError("bsm_max_function_evaluations must be positive")
        if self.comparison_loss not in COMPARISON_LOSSES:
            raise ValueError("comparison_loss is not supported")
        if not isinstance(self.bootstrap_resamples, int) or self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be an integer of at least 100")
        if self.bootstrap_block_length is not None and (
            not isinstance(self.bootstrap_block_length, int) or self.bootstrap_block_length < 1
        ):
            raise ValueError("bootstrap_block_length must be a positive integer or None")
        if not isinstance(self.bootstrap_seed, int) or isinstance(self.bootstrap_seed, bool):
            raise ValueError("bootstrap_seed must be an integer")
        if not 0.5 < self.comparison_confidence_level < 1.0:
            raise ValueError("comparison_confidence_level must lie between 0.5 and 1")
        if not isinstance(self.minimum_comparison_dates, int) or self.minimum_comparison_dates < 3:
            raise ValueError("minimum_comparison_dates must be an integer of at least three")
        if self.comparison_hac_lags is not None and (
            not isinstance(self.comparison_hac_lags, int)
            or isinstance(self.comparison_hac_lags, bool)
            or self.comparison_hac_lags < 0
        ):
            raise ValueError("comparison_hac_lags must be a non-negative integer or None")


@dataclass(frozen=True, slots=True)
class SnapshotModelFit:
    """One contemporaneous fit that may train only later forecasts."""

    sequence: int
    timestamp: str
    snapshot_hash: str
    quote_rows: int
    all_quotes_valid: bool
    surface_downstream_eligible: bool | None
    heston_calibration_eligible: bool | None
    bsm_volatility: float | None
    bsm_objective_rmse: float | None
    heston_parameters: HestonParameters | None
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkPrediction:
    """One truly out-of-sample model price and its quote-aware losses."""

    evaluation_sequence: int
    evaluation_timestamp: str
    training_start_timestamp: str
    training_end_timestamp: str
    model: str
    stable_contract_id: str | None
    expiration: str
    strike: float
    option_type: str
    maturity_years: float
    log_forward_moneyness: float
    maturity_bucket: str
    moneyness_bucket: str
    bid: float
    ask: float
    midpoint: float
    model_price: float
    price_error: float
    absolute_error: float
    squared_error: float
    residual_scale: float
    vega_scaled_error: float
    absolute_vega_scaled_error: float
    squared_vega_scaled_error: float
    quote_band_loss: float
    squared_quote_band_loss: float
    inside_quote: bool


@dataclass(frozen=True, slots=True)
class BenchmarkMetric:
    """A model-loss summary for one collection of scored quotes."""

    model: str
    quote_count: int
    rmse: float
    mae: float
    bias: float
    vega_scaled_rmse: float
    vega_scaled_mae: float
    quote_coverage: float
    mean_quote_band_loss: float
    root_mean_square_quote_band_loss: float


@dataclass(frozen=True, slots=True)
class ModelComparisonInference:
    """Date-clustered bootstrap inference plus paired forecast-test diagnostics."""

    loss: str
    method: str
    aggregation: str
    evaluation_dates: int
    paired_quote_rows: int
    resamples: int
    block_length: int | None
    seed: int
    confidence_level: float
    mean_loss_difference: float | None
    bootstrap_standard_error: float | None
    confidence_interval_low: float | None
    confidence_interval_high: float | None
    probability_heston_better: float | None
    dm_method: str
    hac_lags: int | None
    dm_statistic: float | None
    dm_p_value: float | None
    dm_rejects_equal_accuracy: bool
    dm_available: bool
    dm_failure_reason: str | None
    sign_test_method: str
    heston_win_dates: int
    bsm_win_dates: int
    tie_dates: int
    sign_test_p_value: float | None
    sign_test_rejects_equal_win_rate: bool
    sign_test_available: bool
    sign_test_failure_reason: str | None
    preferred_model: str | None
    conclusive: bool
    available: bool
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkStepAudit:
    """Hash-chained audit for one chronological input snapshot."""

    sequence: int
    timestamp: str
    action: str
    snapshot_hash: str
    previous_audit_hash: str | None
    audit_hash: str
    quote_rows: int
    all_quotes_valid: bool
    training_start_timestamp: str | None
    training_end_timestamp: str | None
    no_lookahead_passed: bool
    bsm_forecast_volatility: float | None
    heston_forecast_parameters: HestonParameters | None
    prediction_rows: int
    prediction_hash: str | None
    bsm_rmse: float | None
    heston_rmse: float | None
    bsm_quote_coverage: float | None
    heston_quote_coverage: float | None
    fit_bsm_volatility: float | None
    fit_heston_parameters: HestonParameters | None
    surface_downstream_eligible: bool | None
    heston_calibration_eligible: bool | None
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChronologicalBenchmarkResult:
    """Rolling out-of-sample comparison with model fits, losses, and audit chain."""

    settings: BenchmarkSettings
    input_snapshot_count: int
    fits: tuple[SnapshotModelFit, ...]
    predictions: tuple[BenchmarkPrediction, ...]
    steps: tuple[BenchmarkStepAudit, ...]
    completed: bool
    downstream_eligible: bool
    evaluation_snapshot_count: int
    final_audit_hash: str | None
    failure_reasons: tuple[str, ...]

    def comparison_inference(self) -> ModelComparisonInference:
        """Compare paired BSM and Heston losses without treating quotes as independent."""
        return model_comparison_inference(self.prediction_frame(), settings=self.settings)

    def prediction_frame(self) -> pd.DataFrame:
        """Return every scored model/contract observation."""
        return pd.DataFrame(asdict(prediction) for prediction in self.predictions)

    def fit_frame(self) -> pd.DataFrame:
        """Return contemporaneous model fits as a flat table."""
        rows: list[dict[str, Any]] = []
        for fit in self.fits:
            row = asdict(fit)
            parameters = row.pop("heston_parameters")
            if parameters is not None:
                row.update({f"heston_{name}": value for name, value in parameters.items()})
            row["failure_reasons"] = "; ".join(row["failure_reasons"])
            rows.append(row)
        return pd.DataFrame(rows)

    def audit_frame(self) -> pd.DataFrame:
        """Return chronological audit steps as a flat table."""
        rows: list[dict[str, Any]] = []
        for step in self.steps:
            row = asdict(step)
            forecast = row.pop("heston_forecast_parameters")
            fitted = row.pop("fit_heston_parameters")
            if forecast is not None:
                row.update({f"forecast_heston_{name}": value for name, value in forecast.items()})
            if fitted is not None:
                row.update({f"fit_heston_{name}": value for name, value in fitted.items()})
            row["failure_reasons"] = "; ".join(row["failure_reasons"])
            rows.append(row)
        return pd.DataFrame(rows)

    def summary_frame(self) -> pd.DataFrame:
        """Return overall loss metrics by model."""
        return _metric_frame(self.prediction_frame())

    def timeline_frame(self) -> pd.DataFrame:
        """Return loss metrics by evaluation snapshot and model."""
        return _metric_frame(self.prediction_frame(), ("evaluation_timestamp",))

    def panel_frame(self) -> pd.DataFrame:
        """Return loss metrics by maturity, moneyness, and model."""
        return _metric_frame(
            self.prediction_frame(),
            ("maturity_bucket", "moneyness_bucket"),
        )


_DEFAULT_BENCHMARK_SETTINGS = BenchmarkSettings()


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return str(value)


def _audit_digest(previous_hash: str | None, fields: dict[str, Any]) -> str:
    payload = json.dumps(
        {"previous_audit_hash": previous_hash, "record": _json_value(fields)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _make_step(**fields: Any) -> BenchmarkStepAudit:
    previous_hash = fields["previous_audit_hash"]
    return BenchmarkStepAudit(audit_hash=_audit_digest(previous_hash, fields), **fields)


def _prediction_digest(predictions: list[BenchmarkPrediction]) -> str:
    payload = json.dumps(
        [_json_value(asdict(prediction)) for prediction in predictions],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metric_from_frame(frame: pd.DataFrame, model: str) -> BenchmarkMetric:
    error = frame["price_error"].to_numpy(dtype=float)
    scaled_error = frame["vega_scaled_error"].to_numpy(dtype=float)
    band_loss = frame["quote_band_loss"].to_numpy(dtype=float)
    return BenchmarkMetric(
        model=model,
        quote_count=len(frame),
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))),
        bias=float(np.mean(error)),
        vega_scaled_rmse=float(np.sqrt(np.mean(scaled_error**2))),
        vega_scaled_mae=float(np.mean(np.abs(scaled_error))),
        quote_coverage=float(frame["inside_quote"].mean()),
        mean_quote_band_loss=float(np.mean(band_loss)),
        root_mean_square_quote_band_loss=float(np.sqrt(np.mean(band_loss**2))),
    )


def _metric_frame(frame: pd.DataFrame, groups: tuple[str, ...] = ()) -> pd.DataFrame:
    columns = [
        *groups,
        "model",
        "quote_count",
        "rmse",
        "mae",
        "bias",
        "vega_scaled_rmse",
        "vega_scaled_mae",
        "quote_coverage",
        "mean_quote_band_loss",
        "root_mean_square_quote_band_loss",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    group_columns = [*groups, "model"]
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in frame.groupby(grouper, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        values = dict(zip(group_columns, key_tuple, strict=True))
        metric = _metric_from_frame(group, str(values["model"]))
        rows.append({**values, **asdict(metric)})
    return pd.DataFrame(rows, columns=columns)


def _comparison_pairs(predictions: pd.DataFrame, loss: str) -> pd.DataFrame:
    """Return one paired BSM/Heston loss difference for each quoted contract."""
    required = {
        "evaluation_sequence",
        "evaluation_timestamp",
        "expiration",
        "strike",
        "option_type",
        "model",
        loss,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"prediction panel is missing comparison columns: {', '.join(missing)}")
    identifiers = [
        "evaluation_sequence",
        "evaluation_timestamp",
        "expiration",
        "strike",
        "option_type",
    ]
    work = predictions.loc[
        predictions["model"].isin(["bsm", "heston"]), [*identifiers, "model", loss]
    ]
    if work.empty:
        return pd.DataFrame(columns=[*identifiers, "loss_difference"])
    if work.duplicated([*identifiers, "model"]).any():
        raise ValueError("prediction panel contains duplicate model losses for a quoted contract")
    paired = work.pivot(index=identifiers, columns="model", values=loss).reset_index()
    if "bsm" not in paired or "heston" not in paired:
        raise ValueError("prediction panel must contain paired BSM and Heston losses")
    if paired[["bsm", "heston"]].isna().any().any():
        raise ValueError("prediction panel contains an unpaired BSM or Heston loss")
    paired["loss_difference"] = paired["heston"] - paired["bsm"]
    return paired


def _diebold_mariano_hac(
    values: np.ndarray,
    *,
    hac_lags: int,
    confidence_level: float,
) -> tuple[float | None, float | None, bool, str | None]:
    """Return an HLN-corrected one-step DM test using a Bartlett HAC variance."""
    count = len(values)
    centered = values - np.mean(values)
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, hac_lags + 1):
        weight = 1.0 - lag / (hac_lags + 1.0)
        autocovariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        long_run_variance += 2.0 * weight * autocovariance
    variance_tolerance = np.finfo(float).eps * max(1.0, float(np.mean(np.square(centered))))
    if not math.isfinite(long_run_variance) or long_run_variance <= variance_tolerance:
        return None, None, False, "HAC long-run variance is zero or numerically invalid"
    standard_error = math.sqrt(long_run_variance / count)
    one_step_hln_correction = math.sqrt((count - 1.0) / count)
    statistic = float(np.mean(values) / standard_error * one_step_hln_correction)
    p_value = float(2.0 * t.sf(abs(statistic), df=count - 1))
    return (
        statistic,
        p_value,
        p_value < 1.0 - confidence_level,
        None,
    )


def model_comparison_inference(
    predictions: pd.DataFrame,
    *,
    settings: BenchmarkSettings = _DEFAULT_BENCHMARK_SETTINGS,
) -> ModelComparisonInference:
    """Estimate uncertainty in mean Heston-minus-BSM loss by circular block bootstrap.

    Contract-level paired losses are averaged inside each evaluation timestamp first. Circular
    blocks then resample that chronological date series, retaining each date's cross section and
    short-range serial dependence.
    """
    method = "circular-moving-block-bootstrap"
    dm_method = "hln-corrected-diebold-mariano-bartlett-hac"
    sign_test_method = "two-sided-exact-binomial-date-sign-test"
    aggregation = "paired-contract-loss-mean-by-evaluation-timestamp"
    paired = _comparison_pairs(predictions, settings.comparison_loss)
    daily = (
        paired.groupby(["evaluation_sequence", "evaluation_timestamp"], sort=True)[
            "loss_difference"
        ]
        .mean()
        .reset_index()
    )
    date_count = len(daily)
    block_length = settings.bootstrap_block_length
    if block_length is None and date_count:
        block_length = max(1, round(date_count ** (1.0 / 3.0)))
    hac_lags = settings.comparison_hac_lags
    if hac_lags is None and date_count:
        hac_lags = max(0, (block_length or 1) - 1)
    values = daily["loss_difference"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("comparison losses must be finite")
    heston_win_dates = int(np.sum(values < 0.0))
    bsm_win_dates = int(np.sum(values > 0.0))
    tie_dates = int(np.sum(values == 0.0))
    if date_count < settings.minimum_comparison_dates:
        insufficient_dates = (
            f"requires at least {settings.minimum_comparison_dates} evaluation dates; "
            f"observed {date_count}"
        )
        return ModelComparisonInference(
            loss=settings.comparison_loss,
            method=method,
            aggregation=aggregation,
            evaluation_dates=date_count,
            paired_quote_rows=len(paired),
            resamples=settings.bootstrap_resamples,
            block_length=block_length,
            seed=settings.bootstrap_seed,
            confidence_level=settings.comparison_confidence_level,
            mean_loss_difference=None,
            bootstrap_standard_error=None,
            confidence_interval_low=None,
            confidence_interval_high=None,
            probability_heston_better=None,
            dm_method=dm_method,
            hac_lags=hac_lags,
            dm_statistic=None,
            dm_p_value=None,
            dm_rejects_equal_accuracy=False,
            dm_available=False,
            dm_failure_reason=insufficient_dates,
            sign_test_method=sign_test_method,
            heston_win_dates=heston_win_dates,
            bsm_win_dates=bsm_win_dates,
            tie_dates=tie_dates,
            sign_test_p_value=None,
            sign_test_rejects_equal_win_rate=False,
            sign_test_available=False,
            sign_test_failure_reason=insufficient_dates,
            preferred_model=None,
            conclusive=False,
            available=False,
            failure_reason=insufficient_dates,
        )
    if block_length is None or block_length > date_count:
        raise ValueError("bootstrap_block_length cannot exceed the evaluation-date count")
    if hac_lags is None or hac_lags >= date_count:
        raise ValueError("comparison_hac_lags must be smaller than the evaluation-date count")
    rng = np.random.default_rng(settings.bootstrap_seed)
    block_count = math.ceil(date_count / block_length)
    starts = rng.integers(0, date_count, size=(settings.bootstrap_resamples, block_count))
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % date_count
    bootstrap_means = values[
        indices.reshape(settings.bootstrap_resamples, -1)[:, :date_count]
    ].mean(axis=1)
    alpha = 1.0 - settings.comparison_confidence_level
    low, high = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    difference = float(np.mean(values))
    preferred = "heston" if difference < 0.0 else "bsm" if difference > 0.0 else "tie"
    dm_statistic, dm_p_value, dm_rejects, dm_failure = _diebold_mariano_hac(
        values,
        hac_lags=hac_lags,
        confidence_level=settings.comparison_confidence_level,
    )
    non_tie_dates = heston_win_dates + bsm_win_dates
    sign_available = non_tie_dates >= settings.minimum_comparison_dates
    sign_failure = (
        None
        if sign_available
        else (
            f"requires at least {settings.minimum_comparison_dates} non-tied evaluation dates; "
            f"observed {non_tie_dates}"
        )
    )
    sign_p_value = (
        float(binomtest(heston_win_dates, non_tie_dates, p=0.5).pvalue) if sign_available else None
    )
    return ModelComparisonInference(
        loss=settings.comparison_loss,
        method=method,
        aggregation=aggregation,
        evaluation_dates=date_count,
        paired_quote_rows=len(paired),
        resamples=settings.bootstrap_resamples,
        block_length=block_length,
        seed=settings.bootstrap_seed,
        confidence_level=settings.comparison_confidence_level,
        mean_loss_difference=difference,
        bootstrap_standard_error=float(np.std(bootstrap_means, ddof=1)),
        confidence_interval_low=float(low),
        confidence_interval_high=float(high),
        probability_heston_better=float(np.mean(bootstrap_means < 0.0)),
        dm_method=dm_method,
        hac_lags=hac_lags,
        dm_statistic=dm_statistic,
        dm_p_value=dm_p_value,
        dm_rejects_equal_accuracy=dm_rejects,
        dm_available=dm_failure is None,
        dm_failure_reason=dm_failure,
        sign_test_method=sign_test_method,
        heston_win_dates=heston_win_dates,
        bsm_win_dates=bsm_win_dates,
        tie_dates=tie_dates,
        sign_test_p_value=sign_p_value,
        sign_test_rejects_equal_win_rate=bool(sign_p_value is not None and sign_p_value < alpha),
        sign_test_available=sign_available,
        sign_test_failure_reason=sign_failure,
        preferred_model=preferred,
        conclusive=bool(high < 0.0 or low > 0.0),
        available=True,
        failure_reason=None,
    )


def _fit_bsm_volatility(
    chain: pd.DataFrame,
    settings: BenchmarkSettings,
) -> tuple[float, float]:
    rows = chain.loc[chain["is_valid"] & chain["iv_converged"]]
    if rows.empty:
        raise ValueError("BSM calibration requires valid implied-volatility observations")
    observed_volatility = rows["implied_volatility"].to_numpy(dtype=float)
    observed_price = rows["midpoint"].to_numpy(dtype=float)
    scales = np.asarray(
        [
            max(
                black_scholes.greeks(
                    str(row.option_type),
                    float(row.underlying_spot),
                    float(row.strike),
                    float(row.maturity_years),
                    float(row.implied_volatility),
                    float(row.risk_free_rate),
                    float(row.dividend_yield),
                ).vega,
                0.02,
            )
            for row in rows.itertuples(index=False)
        ]
    )

    def residuals(values: np.ndarray) -> np.ndarray:
        volatility = float(values[0])
        prices = np.asarray(
            [
                black_scholes.price(
                    str(row.option_type),
                    float(row.underlying_spot),
                    float(row.strike),
                    float(row.maturity_years),
                    volatility,
                    float(row.risk_free_rate),
                    float(row.dividend_yield),
                )
                for row in rows.itertuples(index=False)
            ]
        )
        return (prices - observed_price) / scales

    initial = float(
        np.clip(
            np.median(observed_volatility),
            settings.bsm_volatility_min,
            settings.bsm_volatility_max,
        )
    )
    result = least_squares(
        residuals,
        np.asarray([initial]),
        bounds=(
            np.asarray([settings.bsm_volatility_min]),
            np.asarray([settings.bsm_volatility_max]),
        ),
        max_nfev=settings.bsm_max_function_evaluations,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    if not result.success:
        raise RuntimeError(f"BSM optimizer failed: {result.message}")
    fitted = residuals(np.asarray(result.x, dtype=float))
    return float(result.x[0]), float(np.sqrt(np.mean(fitted**2)))


def _fit_snapshot(
    sequence: int,
    timestamp: pd.Timestamp,
    snapshot: pd.DataFrame,
    enriched: pd.DataFrame,
    settings: BenchmarkSettings,
    diagnostic_grid: np.ndarray,
    heston_grid: np.ndarray,
    cos_settings: COSSettings,
) -> SnapshotModelFit:
    reasons: list[str] = []
    all_valid = bool(enriched["is_valid"].all())
    bsm_volatility: float | None = None
    bsm_rmse: float | None = None
    surface_eligible: bool | None = None
    heston_eligible: bool | None = None
    parameters: HestonParameters | None = None
    if not all_valid:
        reasons.append(
            f"snapshot contains {int((~enriched['is_valid']).sum())} invalid quote row(s)"
        )
    if not reasons:
        try:
            bsm_volatility, bsm_rmse = _fit_bsm_volatility(enriched, settings)
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            reasons.append(f"BSM calibration failed: {exc}")
    surface = None
    if not reasons:
        try:
            quotes = prepare_surface_quotes(enriched)
            surface = (
                calibrate_svi_surface(quotes, diagnostic_grid=diagnostic_grid)
                if settings.surface_model == "svi"
                else calibrate_ssvi(quotes, diagnostic_grid=diagnostic_grid)
            )
            surface_eligible = surface.downstream_eligible
            if not surface_eligible:
                reasons.extend(surface.failure_reasons)
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            surface_eligible = False
            reasons.append(f"surface calibration failed: {exc}")
    if not reasons and surface is not None:
        try:
            calibration = calibrate_heston(
                surface,
                log_moneyness=heston_grid,
                settings=cos_settings,
                require_feller=settings.require_feller,
                max_function_evaluations=settings.heston_max_function_evaluations,
            )
            heston_eligible = calibration.calibration_eligible
            parameters = calibration.parameters
            if not heston_eligible:
                reasons.extend(calibration.failure_reasons)
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            heston_eligible = False
            reasons.append(f"Heston calibration failed: {exc}")
    return SnapshotModelFit(
        sequence=sequence,
        timestamp=timestamp.isoformat(),
        snapshot_hash=snapshot_sha256(snapshot),
        quote_rows=len(snapshot),
        all_quotes_valid=all_valid,
        surface_downstream_eligible=surface_eligible,
        heston_calibration_eligible=heston_eligible,
        bsm_volatility=bsm_volatility,
        bsm_objective_rmse=bsm_rmse,
        heston_parameters=parameters,
        failure_reasons=tuple(reasons),
    )


def _aggregate_parameters(
    fits: list[SnapshotModelFit],
    require_feller: bool,
) -> tuple[float, HestonParameters]:
    if any(fit.bsm_volatility is None or fit.heston_parameters is None for fit in fits):
        raise RuntimeError("training window contains an incomplete model fit")
    bsm_volatility = float(np.median([fit.bsm_volatility for fit in fits]))
    parameter_matrix = np.asarray(
        [fit.heston_parameters.as_array() for fit in fits if fit.heston_parameters is not None]
    )
    parameters = HestonParameters.from_array(np.median(parameter_matrix, axis=0))
    if require_feller and not parameters.feller_satisfied:
        raise ValueError("median rolling Heston parameters violate the required Feller condition")
    return bsm_volatility, parameters


def _maturity_bucket(maturity: float) -> str:
    days = maturity * 365.0
    if days < 60.0:
        return "short (<60d)"
    if days < 180.0:
        return "medium (60-180d)"
    return "long (>=180d)"


def _moneyness_bucket(log_forward_moneyness: float) -> str:
    ratio = math.exp(-log_forward_moneyness)
    if ratio < 0.97:
        return "high strike"
    if ratio > 1.03:
        return "low strike"
    return "near forward"


def _score_snapshot(
    sequence: int,
    timestamp: pd.Timestamp,
    enriched: pd.DataFrame,
    training_fits: list[SnapshotModelFit],
    bsm_volatility: float,
    heston_parameters: HestonParameters,
    cos_settings: COSSettings,
) -> list[BenchmarkPrediction]:
    prices: dict[str, np.ndarray] = {
        "bsm": np.empty(len(enriched), dtype=float),
        "heston": np.empty(len(enriched), dtype=float),
    }
    for position, row in enumerate(enriched.itertuples(index=False)):
        prices["bsm"][position] = black_scholes.price(
            str(row.option_type),
            float(row.underlying_spot),
            float(row.strike),
            float(row.maturity_years),
            bsm_volatility,
            float(row.risk_free_rate),
            float(row.dividend_yield),
        )
    grouping = enriched.assign(_position=np.arange(len(enriched))).groupby(
        ["option_type", "maturity_years", "underlying_spot", "risk_free_rate", "dividend_yield"],
        sort=False,
    )
    for keys, group in grouping:
        option_type, maturity, spot, rate, dividend_yield = keys
        positions = group["_position"].to_numpy(dtype=int)
        prices["heston"][positions] = cos_prices(
            str(option_type),
            float(spot),
            group["strike"].to_numpy(dtype=float),
            float(maturity),
            heston_parameters,
            float(rate),
            float(dividend_yield),
            settings=cos_settings,
        )
    predictions: list[BenchmarkPrediction] = []
    training_start = training_fits[0].timestamp
    training_end = training_fits[-1].timestamp
    for position, row in enumerate(enriched.itertuples(index=False)):
        maturity = float(row.maturity_years)
        forward = float(row.underlying_spot) * math.exp(
            (float(row.risk_free_rate) - float(row.dividend_yield)) * maturity
        )
        log_moneyness = math.log(float(row.strike) / forward)
        bid = float(row.bid)
        ask = float(row.ask)
        midpoint = float(row.midpoint)
        residual_scale = max(
            black_scholes.greeks(
                str(row.option_type),
                float(row.underlying_spot),
                float(row.strike),
                maturity,
                float(row.implied_volatility),
                float(row.risk_free_rate),
                float(row.dividend_yield),
            ).vega,
            0.02,
        )
        stable_contract_id = getattr(row, "stable_contract_id", None)
        if stable_contract_id is not None and pd.isna(stable_contract_id):
            stable_contract_id = None
        for model in ("bsm", "heston"):
            model_price = float(prices[model][position])
            error = model_price - midpoint
            scaled_error = error / residual_scale
            band_loss = max(bid - model_price, model_price - ask, 0.0)
            predictions.append(
                BenchmarkPrediction(
                    evaluation_sequence=sequence,
                    evaluation_timestamp=timestamp.isoformat(),
                    training_start_timestamp=training_start,
                    training_end_timestamp=training_end,
                    model=model,
                    stable_contract_id=(
                        str(stable_contract_id) if stable_contract_id is not None else None
                    ),
                    expiration=pd.Timestamp(row.expiration).isoformat(),
                    strike=float(row.strike),
                    option_type=str(row.option_type),
                    maturity_years=maturity,
                    log_forward_moneyness=log_moneyness,
                    maturity_bucket=_maturity_bucket(maturity),
                    moneyness_bucket=_moneyness_bucket(log_moneyness),
                    bid=bid,
                    ask=ask,
                    midpoint=midpoint,
                    model_price=model_price,
                    price_error=error,
                    absolute_error=abs(error),
                    squared_error=error**2,
                    residual_scale=residual_scale,
                    vega_scaled_error=scaled_error,
                    absolute_vega_scaled_error=abs(scaled_error),
                    squared_vega_scaled_error=scaled_error**2,
                    quote_band_loss=band_loss,
                    squared_quote_band_loss=band_loss**2,
                    inside_quote=bid <= model_price <= ask,
                )
            )
    return predictions


def verify_benchmark_hash_chain(result: ChronologicalBenchmarkResult) -> bool:
    """Verify chronological linkage plus every scored prediction panel hash."""
    predictions_by_sequence: dict[int, list[BenchmarkPrediction]] = {}
    for prediction in result.predictions:
        predictions_by_sequence.setdefault(prediction.evaluation_sequence, []).append(prediction)
    previous: str | None = None
    for expected_sequence, step in enumerate(result.steps):
        if step.sequence != expected_sequence or step.previous_audit_hash != previous:
            return False
        fields = asdict(step)
        observed = str(fields.pop("audit_hash"))
        if _audit_digest(previous, fields) != observed:
            return False
        step_predictions = predictions_by_sequence.get(step.sequence, [])
        if step.prediction_rows != len(step_predictions):
            return False
        expected_prediction_hash = (
            _prediction_digest(step_predictions) if step_predictions else None
        )
        if step.prediction_hash != expected_prediction_hash:
            return False
        previous = observed
    expected_prediction_count = sum(step.prediction_rows for step in result.steps)
    return (
        expected_prediction_count == len(result.predictions) and result.final_audit_hash == previous
    )


def run_chronological_benchmark(
    frame: pd.DataFrame,
    *,
    settings: BenchmarkSettings = _DEFAULT_BENCHMARK_SETTINGS,
) -> ChronologicalBenchmarkResult:
    """Fit trailing snapshots and score only the untouched next snapshot."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ChainSchemaError(f"missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("benchmark input must contain at least one quote")
    work = frame.copy(deep=True)
    if "reference_validated" in work:
        reference_passed = work["reference_validated"].map(
            lambda value: value is True or str(value).strip().lower() == "true"
        )
        if not reference_passed.all():
            raise ValueError(
                "benchmark input contains rows that failed contract/session reference validation"
            )
        if "stable_contract_id" not in work:
            raise ChainSchemaError(
                "reference-validated benchmark input requires stable_contract_id"
            )
    work["_parsed_timestamp"] = pd.to_datetime(work["quote_timestamp"], errors="coerce", utc=True)
    if work["_parsed_timestamp"].isna().any():
        raise ValueError("every benchmark quote_timestamp must be valid")
    timestamps = tuple(sorted(work["_parsed_timestamp"].unique()))
    if len(timestamps) <= settings.training_window:
        raise ValueError("benchmark requires more snapshots than the rolling training window")

    fits: list[SnapshotModelFit] = []
    predictions: list[BenchmarkPrediction] = []
    steps: list[BenchmarkStepAudit] = []
    failures: list[str] = []
    previous_hash: str | None = None
    diagnostic_grid = np.linspace(-1.0, 1.0, settings.surface_grid_points)
    heston_grid = np.linspace(
        settings.heston_moneyness_min,
        settings.heston_moneyness_max,
        settings.heston_moneyness_points,
    )
    cos_settings = COSSettings(settings.cos_terms, settings.cos_truncation_width)

    for sequence, raw_timestamp in enumerate(timestamps):
        timestamp = pd.Timestamp(raw_timestamp)
        snapshot = work.loc[work["_parsed_timestamp"] == raw_timestamp].drop(
            columns="_parsed_timestamp"
        )
        content_hash = snapshot_sha256(snapshot)
        enriched = enrich_option_chain(
            snapshot,
            model="bsm",
            valuation_time=timestamp,
            stale_after="0s",
        )
        all_valid = bool(enriched["is_valid"].all())
        step_reasons: list[str] = []
        window: list[SnapshotModelFit] = []
        bsm_forecast: float | None = None
        heston_forecast: HestonParameters | None = None
        step_predictions: list[BenchmarkPrediction] = []
        bsm_metric: BenchmarkMetric | None = None
        heston_metric: BenchmarkMetric | None = None
        no_lookahead = True

        if not all_valid:
            step_reasons.append(
                f"snapshot contains {int((~enriched['is_valid']).sum())} invalid quote row(s)"
            )
        if not step_reasons and sequence >= settings.training_window:
            window = fits[-settings.training_window :]
            no_lookahead = bool(
                len(window) == settings.training_window
                and pd.Timestamp(window[-1].timestamp) < timestamp
            )
            if not no_lookahead:
                step_reasons.append("training window does not end before evaluation timestamp")
            if not step_reasons:
                try:
                    bsm_forecast, heston_forecast = _aggregate_parameters(
                        window, settings.require_feller
                    )
                    step_predictions = _score_snapshot(
                        sequence,
                        timestamp,
                        enriched,
                        window,
                        bsm_forecast,
                        heston_forecast,
                        cos_settings,
                    )
                    prediction_frame = pd.DataFrame(
                        asdict(prediction) for prediction in step_predictions
                    )
                    bsm_metric = _metric_from_frame(
                        prediction_frame.loc[prediction_frame["model"] == "bsm"], "bsm"
                    )
                    heston_metric = _metric_from_frame(
                        prediction_frame.loc[prediction_frame["model"] == "heston"], "heston"
                    )
                except (ArithmeticError, RuntimeError, ValueError) as exc:
                    step_reasons.append(f"out-of-sample scoring failed: {exc}")

        fit: SnapshotModelFit | None = None
        if not step_reasons and sequence < len(timestamps) - 1:
            fit = _fit_snapshot(
                sequence,
                timestamp,
                snapshot,
                enriched,
                settings,
                diagnostic_grid,
                heston_grid,
                cos_settings,
            )
            fits.append(fit)
            step_reasons.extend(fit.failure_reasons)

        action = (
            "blocked"
            if step_reasons
            else "score"
            if sequence == len(timestamps) - 1
            else "score_and_fit"
            if step_predictions
            else "fit_only"
        )
        if step_predictions and not step_reasons:
            predictions.extend(step_predictions)
        step = _make_step(
            sequence=sequence,
            timestamp=timestamp.isoformat(),
            action=action,
            snapshot_hash=content_hash,
            previous_audit_hash=previous_hash,
            quote_rows=len(snapshot),
            all_quotes_valid=all_valid,
            training_start_timestamp=window[0].timestamp if window else None,
            training_end_timestamp=window[-1].timestamp if window else None,
            no_lookahead_passed=no_lookahead,
            bsm_forecast_volatility=bsm_forecast,
            heston_forecast_parameters=heston_forecast,
            prediction_rows=len(step_predictions) if not step_reasons else 0,
            prediction_hash=(
                _prediction_digest(step_predictions)
                if step_predictions and not step_reasons
                else None
            ),
            bsm_rmse=bsm_metric.rmse if bsm_metric is not None and not step_reasons else None,
            heston_rmse=(
                heston_metric.rmse if heston_metric is not None and not step_reasons else None
            ),
            bsm_quote_coverage=(
                bsm_metric.quote_coverage if bsm_metric is not None and not step_reasons else None
            ),
            heston_quote_coverage=(
                heston_metric.quote_coverage
                if heston_metric is not None and not step_reasons
                else None
            ),
            fit_bsm_volatility=fit.bsm_volatility if fit is not None else None,
            fit_heston_parameters=fit.heston_parameters if fit is not None else None,
            surface_downstream_eligible=(
                fit.surface_downstream_eligible if fit is not None else None
            ),
            heston_calibration_eligible=(
                fit.heston_calibration_eligible if fit is not None else None
            ),
            failure_reasons=tuple(step_reasons),
        )
        steps.append(step)
        previous_hash = step.audit_hash
        if step_reasons:
            failures.extend(
                f"snapshot {sequence} ({timestamp.isoformat()}): {reason}"
                for reason in step_reasons
            )
            break

    evaluation_count = len({prediction.evaluation_sequence for prediction in predictions})
    expected_evaluations = len(timestamps) - settings.training_window
    completed = (
        len(steps) == len(timestamps)
        and not failures
        and evaluation_count == expected_evaluations
        and bool(predictions)
    )
    result = ChronologicalBenchmarkResult(
        settings=settings,
        input_snapshot_count=len(timestamps),
        fits=tuple(fits),
        predictions=tuple(predictions),
        steps=tuple(steps),
        completed=completed,
        downstream_eligible=False,
        evaluation_snapshot_count=evaluation_count,
        final_audit_hash=previous_hash,
        failure_reasons=tuple(failures),
    )
    chain_verified = verify_benchmark_hash_chain(result)
    if not chain_verified:
        result = replace(
            result,
            failure_reasons=(
                *result.failure_reasons,
                "benchmark audit hash-chain verification failed",
            ),
        )
    return replace(result, downstream_eligible=completed and chain_verified)


def benchmark_summary(result: ChronologicalBenchmarkResult) -> dict[str, Any]:
    """Return a deterministic JSON-compatible benchmark audit."""
    return {
        "engine": "rolling-bsm-heston-benchmark",
        "settings": asdict(result.settings),
        "input_snapshot_count": result.input_snapshot_count,
        "processed_snapshot_count": len(result.steps),
        "evaluation_snapshot_count": result.evaluation_snapshot_count,
        "prediction_count": len(result.predictions),
        "completed": result.completed,
        "downstream_eligible": result.downstream_eligible,
        "final_audit_hash": result.final_audit_hash,
        "hash_chain_verified": verify_benchmark_hash_chain(result),
        "failure_reasons": list(result.failure_reasons),
        "comparison_inference": asdict(result.comparison_inference()),
        "overall_metrics": result.summary_frame().to_dict(orient="records"),
        "maturity_moneyness_panels": result.panel_frame().to_dict(orient="records"),
        "fits": [asdict(fit) for fit in result.fits],
        "steps": [asdict(step) for step in result.steps],
    }


def benchmark_report_json(result: ChronologicalBenchmarkResult) -> bytes:
    """Serialize a deterministic human-readable benchmark audit."""
    return (json.dumps(benchmark_summary(result), indent=2, sort_keys=True) + "\n").encode("utf-8")
