"""Pre-expiry correlated-GBM pricing for explicit option deliverables."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from volforge.deliverables import MAX_DELIVERABLE_BYTES, DeliverableTerms
from volforge.models.common import OptionType

MAX_BASKET_ASSETS = 16


@dataclass(frozen=True, slots=True)
class BasketAssetMarket:
    """Constant risk-neutral GBM inputs for one basket component."""

    asset_id: str
    spot: float
    volatility: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        asset_id = str(self.asset_id).strip()
        if not asset_id:
            raise ValueError("basket asset_id must be non-empty")
        spot = _finite(self.spot, f"spot for {asset_id}")
        volatility = _finite(self.volatility, f"volatility for {asset_id}")
        dividend_yield = _finite(self.dividend_yield, f"dividend_yield for {asset_id}")
        if spot <= 0.0:
            raise ValueError(f"spot for {asset_id} must be positive")
        if volatility < 0.0:
            raise ValueError(f"volatility for {asset_id} must be non-negative")
        object.__setattr__(self, "asset_id", asset_id)
        object.__setattr__(self, "spot", spot)
        object.__setattr__(self, "volatility", volatility)
        object.__setattr__(self, "dividend_yield", dividend_yield)


@dataclass(frozen=True, slots=True)
class BasketMarket:
    """Ordered component markets and their Brownian correlation matrix."""

    assets: tuple[BasketAssetMarket, ...]
    correlation: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        matrix = _validated_correlation(self.assets, self.correlation)
        object.__setattr__(
            self,
            "correlation",
            tuple(tuple(float(value) for value in row) for row in matrix),
        )

    @property
    def asset_ids(self) -> tuple[str, ...]:
        """Return correlation-matrix asset order."""
        return tuple(asset.asset_id for asset in self.assets)

    @property
    def canonical_json(self) -> str:
        """Return deterministic market-input JSON in matrix order."""
        return _canonical_json(
            {
                "assets": [asdict(asset) for asset in self.assets],
                "correlation": [list(row) for row in self.correlation],
            }
        )

    @property
    def sha256(self) -> str:
        """Return the canonical market-input digest."""
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BasketMonteCarloSettings:
    """Sampling and interval controls for basket pricing."""

    paths: int = 100_000
    seed: int = 7
    antithetic: bool = True
    control_variate: bool = True
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if (
            isinstance(self.paths, bool)
            or not isinstance(self.paths, int)
            or not 2 <= self.paths <= 2_000_000
        ):
            raise ValueError("paths must be an integer between 2 and 2,000,000")
        if self.antithetic and self.paths % 2:
            raise ValueError("paths must be even when antithetic sampling is enabled")
        if self.antithetic and self.paths < 4:
            raise ValueError("paths must be at least 4 when antithetic sampling is enabled")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not math.isfinite(self.confidence_level) or not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie between 0.5 and 1")


@dataclass(frozen=True, slots=True)
class BasketDeltaEstimate:
    """Pathwise price sensitivity to one component spot."""

    asset_id: str
    delta: float
    standard_error: float


@dataclass(frozen=True, slots=True)
class BasketPricingResult:
    """Auditable correlated-GBM basket-option price and diagnostics."""

    engine: str
    option_type: str
    settlement_method: str
    classification: str
    currency: str
    strike: float
    contract_multiplier: float
    maturity: float
    rate: float
    paths: int
    independent_samples: int
    seed: int
    antithetic: bool
    control_variate: bool
    confidence_level: float
    price: float
    standard_error: float
    confidence_interval_low: float
    confidence_interval_high: float
    crude_price: float
    crude_standard_error: float
    control_beta: float
    variance_reduction_ratio: float
    moment_matched_price: float
    moment_match_difference: float
    lower_bound: float
    upper_bound: float
    bounds_passed: bool
    expected_discounted_deliverable: float
    risk_factor_ids: tuple[str, ...]
    deltas: tuple[BasketDeltaEstimate, ...]
    deliverable_sha256: str
    market_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready pricing audit."""
        payload = asdict(self)
        payload["risk_factor_ids"] = list(self.risk_factor_ids)
        payload["deltas"] = [asdict(delta) for delta in self.deltas]
        return payload


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


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


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


def _validated_correlation(
    assets: tuple[BasketAssetMarket, ...],
    correlation: tuple[tuple[float, ...], ...],
) -> NDArray[np.float64]:
    count = len(assets)
    if count > MAX_BASKET_ASSETS:
        raise ValueError(f"basket market exceeds {MAX_BASKET_ASSETS} assets")
    asset_ids = [asset.asset_id for asset in assets]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("basket market contains duplicate asset_id values")
    if count == 0 and not correlation:
        return np.empty((0, 0), dtype=float)
    try:
        matrix = np.asarray(correlation, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("correlation must be a finite square matrix") from exc
    if matrix.shape != (count, count):
        raise ValueError(f"correlation must have shape ({count}, {count})")
    if not np.isfinite(matrix).all():
        raise ValueError("correlation must be finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10):
        raise ValueError("correlation must be symmetric")
    if count and not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("correlation diagonal must equal one")
    if np.any(matrix < -1.0 - 1e-10) or np.any(matrix > 1.0 + 1e-10):
        raise ValueError("correlation entries must lie in [-1, 1]")
    if count and float(np.linalg.eigvalsh(matrix).min()) < -1e-10:
        raise ValueError("correlation must be positive semidefinite")
    return matrix


def parse_basket_market(value: object) -> BasketMarket:
    """Parse strict JSON market inputs and canonicalize correlation order."""
    parsed = _json_value(value, "basket_market")
    if not isinstance(parsed, dict):
        raise ValueError("basket_market must be a JSON object")
    unexpected = sorted(set(parsed) - {"assets", "correlation"})
    if unexpected:
        raise ValueError("basket_market has unsupported fields: " + ", ".join(unexpected))
    raw_assets = parsed.get("assets")
    if not isinstance(raw_assets, list):
        raise ValueError("basket_market assets must be a JSON list")
    if len(raw_assets) > MAX_BASKET_ASSETS:
        raise ValueError(f"basket market exceeds {MAX_BASKET_ASSETS} assets")
    assets: list[BasketAssetMarket] = []
    for position, raw_asset in enumerate(raw_assets):
        if not isinstance(raw_asset, dict):
            raise ValueError(f"basket market asset {position} must be an object")
        extra = sorted(set(raw_asset) - {"asset_id", "spot", "volatility", "dividend_yield"})
        if extra:
            raise ValueError(
                f"basket market asset {position} has unsupported fields: {', '.join(extra)}"
            )
        assets.append(
            BasketAssetMarket(
                asset_id=str(raw_asset.get("asset_id", "")),
                spot=raw_asset.get("spot"),  # type: ignore[arg-type]
                volatility=raw_asset.get("volatility"),  # type: ignore[arg-type]
                dividend_yield=raw_asset.get("dividend_yield", 0.0),  # type: ignore[arg-type]
            )
        )
    raw_correlation = parsed.get("correlation")
    if not isinstance(raw_correlation, list):
        raise ValueError("basket_market correlation must be a JSON matrix")
    if any(not isinstance(row, list) for row in raw_correlation):
        raise ValueError("basket_market correlation rows must be JSON lists")
    original = BasketMarket(
        assets=tuple(assets),
        correlation=tuple(tuple(row) for row in raw_correlation),
    )
    order = sorted(range(len(assets)), key=lambda index: assets[index].asset_id)
    if order == list(range(len(assets))):
        return original
    matrix = np.asarray(original.correlation, dtype=float)
    reordered = matrix[np.ix_(order, order)]
    return BasketMarket(
        assets=tuple(assets[index] for index in order),
        correlation=tuple(tuple(float(value) for value in row) for row in reordered),
    )


def _package_exposures(
    terms: DeliverableTerms,
) -> tuple[tuple[str, ...], NDArray[np.float64], float]:
    if terms.settlement_method == "cash":
        if terms.underlying_id is None:
            raise ValueError("cash-settled pre-expiry pricing requires underlying_id")
        return (
            (terms.underlying_id,),
            np.asarray([terms.components[0].quantity], dtype=float),
            0.0,
        )
    securities = [component for component in terms.components if component.asset_type == "security"]
    fixed_cash = sum(
        component.quantity
        for component in terms.components
        if component.asset_type in {"cash", "cash_in_lieu"}
    )
    return (
        tuple(component.asset_id for component in securities),
        np.asarray([component.quantity for component in securities], dtype=float),
        fixed_cash,
    )


def _moment_matched_price(
    kind: OptionType,
    strike_amount: float,
    maturity: float,
    rate: float,
    quantities: NDArray[np.float64],
    fixed_cash: float,
    spots: NDArray[np.float64],
    volatilities: NDArray[np.float64],
    dividends: NDArray[np.float64],
    correlation: NDArray[np.float64],
) -> float:
    discount = math.exp(-rate * maturity)
    forwards = spots * np.exp((rate - dividends) * maturity)
    random_mean = float(np.dot(quantities, forwards))
    total_mean = random_mean + fixed_cash
    if len(spots):
        cross_moments = np.outer(forwards, forwards) * np.exp(
            np.outer(volatilities, volatilities) * correlation * maturity
        )
        random_second = float(quantities @ cross_moments @ quantities)
    else:
        random_second = 0.0
    total_second = random_second + 2.0 * fixed_cash * random_mean + fixed_cash**2
    variance = max(total_second - total_mean**2, 0.0)
    intrinsic = (
        total_mean - strike_amount if kind is OptionType.CALL else strike_amount - total_mean
    )
    if variance <= 1e-24 or total_mean <= 0.0:
        return discount * max(intrinsic, 0.0)
    if strike_amount == 0.0:
        return discount * total_mean if kind is OptionType.CALL else 0.0
    log_variance = math.log1p(variance / total_mean**2)
    log_volatility = math.sqrt(log_variance)
    d2 = (math.log(total_mean / strike_amount) - 0.5 * log_variance) / log_volatility
    d1 = d2 + log_volatility
    call = discount * (total_mean * float(norm.cdf(d1)) - strike_amount * float(norm.cdf(d2)))
    if kind is OptionType.CALL:
        return call
    return call - discount * (total_mean - strike_amount)


def _sample_standard_error(values: NDArray[np.float64]) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _estimator_observations(
    values: NDArray[np.float64], *, antithetic: bool
) -> NDArray[np.float64]:
    """Collapse dependent antithetic paths into independent pair means."""
    if not antithetic:
        return values
    half = len(values) // 2
    return 0.5 * (values[:half] + values[half:])


def price_basket_option(
    terms: DeliverableTerms,
    market: BasketMarket,
    *,
    option_type: str | OptionType,
    strike: float,
    maturity: float,
    rate: float = 0.0,
    settings: BasketMonteCarloSettings | None = None,
) -> BasketPricingResult:
    """Price a European option on explicit deliverables under correlated GBM."""
    kind = OptionType.parse(option_type)
    normalized_strike = _finite(strike, "strike")
    normalized_maturity = _finite(maturity, "maturity")
    normalized_rate = _finite(rate, "rate")
    if normalized_strike < 0.0:
        raise ValueError("strike must be non-negative")
    if normalized_maturity < 0.0:
        raise ValueError("maturity must be non-negative")
    settings = settings or BasketMonteCarloSettings()
    correlation = _validated_correlation(market.assets, market.correlation)
    risk_factor_ids, quantities, fixed_cash = _package_exposures(terms)
    if market.asset_ids != risk_factor_ids:
        missing = sorted(set(risk_factor_ids) - set(market.asset_ids))
        unused = sorted(set(market.asset_ids) - set(risk_factor_ids))
        reasons: list[str] = []
        if missing:
            reasons.append("missing basket assets: " + ", ".join(missing))
        if unused:
            reasons.append("unused basket assets: " + ", ".join(unused))
        if not reasons:
            reasons.append("basket assets must follow canonical risk-factor order")
        raise ValueError("; ".join(reasons))

    spots = np.asarray([asset.spot for asset in market.assets], dtype=float)
    volatilities = np.asarray([asset.volatility for asset in market.assets], dtype=float)
    dividends = np.asarray([asset.dividend_yield for asset in market.assets], dtype=float)
    strike_amount = normalized_strike * terms.contract_multiplier
    discount = math.exp(-normalized_rate * normalized_maturity)
    expected_discounted = float(
        np.dot(quantities, spots * np.exp(-dividends * normalized_maturity)) + fixed_cash * discount
    )
    discounted_strike = strike_amount * discount
    lower_bound = max(
        expected_discounted - discounted_strike
        if kind is OptionType.CALL
        else discounted_strike - expected_discounted,
        0.0,
    )
    upper_bound = expected_discounted if kind is OptionType.CALL else discounted_strike
    moment_price = _moment_matched_price(
        kind,
        strike_amount,
        normalized_maturity,
        normalized_rate,
        quantities,
        fixed_cash,
        spots,
        volatilities,
        dividends,
        correlation,
    )

    count = len(spots)
    if normalized_maturity == 0.0 or count == 0 or not np.any(volatilities > 0.0):
        terminal_assets = spots * np.exp((normalized_rate - dividends) * normalized_maturity)
        delivered = float(np.dot(quantities, terminal_assets) + fixed_cash)
        intrinsic = (
            delivered - strike_amount if kind is OptionType.CALL else strike_amount - delivered
        )
        price = discount * max(intrinsic, 0.0)
        if intrinsic > 0.0:
            indicator = 1.0
        elif intrinsic < 0.0:
            indicator = 0.0
        else:
            indicator = 0.5
        direction = 1.0 if kind is OptionType.CALL else -1.0
        deltas = tuple(
            BasketDeltaEstimate(
                asset_id=asset.asset_id,
                delta=direction
                * indicator
                * discount
                * quantities[index]
                * terminal_assets[index]
                / spots[index],
                standard_error=0.0,
            )
            for index, asset in enumerate(market.assets)
        )
        return BasketPricingResult(
            engine="correlated-gbm-monte-carlo",
            option_type=kind.value,
            settlement_method=terms.settlement_method,
            classification=terms.classification,
            currency=terms.currency,
            strike=normalized_strike,
            contract_multiplier=terms.contract_multiplier,
            maturity=normalized_maturity,
            rate=normalized_rate,
            paths=settings.paths,
            independent_samples=0,
            seed=settings.seed,
            antithetic=settings.antithetic,
            control_variate=settings.control_variate,
            confidence_level=settings.confidence_level,
            price=price,
            standard_error=0.0,
            confidence_interval_low=price,
            confidence_interval_high=price,
            crude_price=price,
            crude_standard_error=0.0,
            control_beta=0.0,
            variance_reduction_ratio=1.0,
            moment_matched_price=moment_price,
            moment_match_difference=price - moment_price,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            bounds_passed=lower_bound - 1e-12 <= price <= upper_bound + 1e-12,
            expected_discounted_deliverable=expected_discounted,
            risk_factor_ids=risk_factor_ids,
            deltas=deltas,
            deliverable_sha256=terms.sha256,
            market_sha256=market.sha256,
        )

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    factor = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))
    rng = np.random.default_rng(settings.seed)
    if settings.antithetic:
        base = rng.standard_normal((settings.paths // 2, count))
        independent = np.concatenate((base, -base), axis=0)
    else:
        independent = rng.standard_normal((settings.paths, count))
    shocks = independent @ factor.T
    terminal = spots * np.exp(
        (normalized_rate - dividends - 0.5 * volatilities**2) * normalized_maturity
        + volatilities * math.sqrt(normalized_maturity) * shocks
    )
    delivered = terminal @ quantities + fixed_cash
    intrinsic = delivered - strike_amount if kind is OptionType.CALL else strike_amount - delivered
    crude_samples = discount * np.maximum(intrinsic, 0.0)
    control_samples = discount * delivered
    crude_observations = _estimator_observations(crude_samples, antithetic=settings.antithetic)
    control_observations = _estimator_observations(control_samples, antithetic=settings.antithetic)
    centered_control = control_observations - expected_discounted
    control_variance = float(np.var(control_observations, ddof=1))
    if settings.control_variate and control_variance > 1e-24:
        beta = float(
            np.cov(crude_observations, control_observations, ddof=1)[0, 1] / control_variance
        )
        price_samples = crude_observations - beta * centered_control
    else:
        beta = 0.0
        price_samples = crude_observations
    price = float(np.mean(price_samples))
    standard_error = _sample_standard_error(price_samples)
    crude_price = float(np.mean(crude_observations))
    crude_standard_error = _sample_standard_error(crude_observations)
    adjusted_variance = float(np.var(price_samples, ddof=1))
    crude_variance = float(np.var(crude_observations, ddof=1))
    variance_reduction = crude_variance / adjusted_variance if adjusted_variance > 1e-24 else 1.0
    quantile = float(norm.ppf(0.5 + 0.5 * settings.confidence_level))
    interval_low = price - quantile * standard_error
    interval_high = price + quantile * standard_error

    active = intrinsic > 0.0
    direction = 1.0 if kind is OptionType.CALL else -1.0
    delta_estimates: list[BasketDeltaEstimate] = []
    for index, asset in enumerate(market.assets):
        terminal_derivative = quantities[index] * terminal[:, index] / spots[index]
        raw_delta = direction * discount * active * terminal_derivative
        raw_delta_observations = _estimator_observations(raw_delta, antithetic=settings.antithetic)
        if beta:
            control_derivative = discount * terminal_derivative
            expected_derivative = quantities[index] * math.exp(
                -asset.dividend_yield * normalized_maturity
            )
            control_delta_observations = _estimator_observations(
                control_derivative, antithetic=settings.antithetic
            )
            delta_samples = raw_delta_observations - beta * (
                control_delta_observations - expected_derivative
            )
        else:
            delta_samples = raw_delta_observations
        delta_estimates.append(
            BasketDeltaEstimate(
                asset_id=asset.asset_id,
                delta=float(np.mean(delta_samples)),
                standard_error=_sample_standard_error(delta_samples),
            )
        )

    tolerance = max(1e-10, 5.0 * standard_error)
    return BasketPricingResult(
        engine="correlated-gbm-monte-carlo",
        option_type=kind.value,
        settlement_method=terms.settlement_method,
        classification=terms.classification,
        currency=terms.currency,
        strike=normalized_strike,
        contract_multiplier=terms.contract_multiplier,
        maturity=normalized_maturity,
        rate=normalized_rate,
        paths=settings.paths,
        independent_samples=len(price_samples),
        seed=settings.seed,
        antithetic=settings.antithetic,
        control_variate=settings.control_variate,
        confidence_level=settings.confidence_level,
        price=price,
        standard_error=standard_error,
        confidence_interval_low=interval_low,
        confidence_interval_high=interval_high,
        crude_price=crude_price,
        crude_standard_error=crude_standard_error,
        control_beta=beta,
        variance_reduction_ratio=variance_reduction,
        moment_matched_price=moment_price,
        moment_match_difference=price - moment_price,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        bounds_passed=lower_bound - tolerance <= price <= upper_bound + tolerance,
        expected_discounted_deliverable=expected_discounted,
        risk_factor_ids=risk_factor_ids,
        deltas=tuple(delta_estimates),
        deliverable_sha256=terms.sha256,
        market_sha256=market.sha256,
    )


__all__ = [
    "MAX_BASKET_ASSETS",
    "BasketAssetMarket",
    "BasketDeltaEstimate",
    "BasketMarket",
    "BasketMonteCarloSettings",
    "BasketPricingResult",
    "parse_basket_market",
    "price_basket_option",
]
