"""Heston stochastic-volatility pricing with the Fourier-cosine method."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volforge.models.common import OptionType, discounted_price_bounds

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class HestonParameters:
    """Risk-neutral Heston variance-process parameters.

    ``kappa`` is mean reversion, ``theta`` long-run variance, ``vol_of_vol``
    the variance diffusion coefficient, ``rho`` the spot/variance Brownian
    correlation, and ``initial_variance`` the time-zero variance.
    """

    kappa: float
    theta: float
    vol_of_vol: float
    rho: float
    initial_variance: float

    def __post_init__(self) -> None:
        values = (
            self.kappa,
            self.theta,
            self.vol_of_vol,
            self.rho,
            self.initial_variance,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Heston parameters must be finite")
        if self.kappa <= 0:
            raise ValueError("Heston kappa must be positive")
        if self.theta <= 0:
            raise ValueError("Heston theta must be positive")
        if self.vol_of_vol <= 0:
            raise ValueError("Heston vol_of_vol must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("Heston rho must be strictly between -1 and 1")
        if self.initial_variance <= 0:
            raise ValueError("Heston initial_variance must be positive")

    @classmethod
    def from_array(cls, values: ArrayLike) -> HestonParameters:
        """Construct parameters in kappa, theta, vol-of-vol, rho, v0 order."""
        array = np.asarray(values, dtype=float)
        if array.shape != (5,):
            raise ValueError("Heston requires exactly five parameters")
        return cls(*(float(value) for value in array))

    def as_array(self) -> FloatArray:
        """Return parameters in kappa, theta, vol-of-vol, rho, v0 order."""
        return np.asarray(
            [self.kappa, self.theta, self.vol_of_vol, self.rho, self.initial_variance],
            dtype=float,
        )

    @property
    def feller_margin(self) -> float:
        """Return ``2*kappa*theta - vol_of_vol**2``."""
        return 2.0 * self.kappa * self.theta - self.vol_of_vol**2

    @property
    def feller_satisfied(self) -> bool:
        """Whether the strict-positivity Feller condition is satisfied."""
        return self.feller_margin >= 0.0


@dataclass(frozen=True, slots=True)
class COSSettings:
    """Numerical settings for Fourier-cosine pricing."""

    terms: int = 256
    truncation_width: float = 16.0

    def __post_init__(self) -> None:
        if not isinstance(self.terms, int) or self.terms < 32:
            raise ValueError("COS terms must be an integer of at least 32")
        if not math.isfinite(self.truncation_width) or not 5.0 <= self.truncation_width <= 30.0:
            raise ValueError("COS truncation_width must be between 5 and 30")


_DEFAULT_COS_SETTINGS = COSSettings()


@dataclass(frozen=True, slots=True)
class HestonPriceResult:
    """A Heston price with numerical and model diagnostics."""

    price: float
    lower_bound: float
    upper_bound: float
    terms: int
    truncation_lower: float
    truncation_upper: float
    feller_margin: float
    feller_satisfied: bool


def heston_log_return_characteristic(
    frequency: ArrayLike,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> ComplexArray:
    """Return the characteristic function of ``log(S_T / S_0)``.

    The implementation uses the numerically stable little-Heston-trap form
    with the principal complex square root and logarithm.
    """
    if not math.isfinite(maturity) or maturity < 0:
        raise ValueError("maturity must be finite and non-negative")
    if not math.isfinite(rate) or not math.isfinite(dividend_yield):
        raise ValueError("rate and dividend_yield must be finite")
    u = np.asarray(frequency, dtype=float)
    if np.any(~np.isfinite(u)):
        raise ValueError("frequency must be finite")
    if maturity == 0:
        return np.ones_like(u, dtype=np.complex128)

    kappa = parameters.kappa
    theta = parameters.theta
    sigma = parameters.vol_of_vol
    rho = parameters.rho
    initial_variance = parameters.initial_variance
    iu = 1j * u
    beta = kappa - rho * sigma * iu
    discriminant = np.sqrt(beta**2 + sigma**2 * (u**2 + iu))
    discriminant = np.where(np.real(discriminant) < 0.0, -discriminant, discriminant)
    g = (beta - discriminant) / (beta + discriminant)
    exponential = np.exp(-discriminant * maturity)
    log_ratio = np.log((1.0 - g * exponential) / (1.0 - g))
    c = iu * (rate - dividend_yield) * maturity + (kappa * theta / sigma**2) * (
        (beta - discriminant) * maturity - 2.0 * log_ratio
    )
    d = ((beta - discriminant) / sigma**2) * ((1.0 - exponential) / (1.0 - g * exponential))
    return np.asarray(np.exp(c + d * initial_variance), dtype=np.complex128)


def heston_cumulants(
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return the first two cumulants of ``log(S_T / S_0)``."""
    if not math.isfinite(maturity) or maturity <= 0:
        raise ValueError("maturity must be positive and finite")
    if not math.isfinite(rate) or not math.isfinite(dividend_yield):
        raise ValueError("rate and dividend_yield must be finite")
    kappa = parameters.kappa
    theta = parameters.theta
    sigma = parameters.vol_of_vol
    rho = parameters.rho
    v0 = parameters.initial_variance
    decay = math.exp(-kappa * maturity)
    c1 = (
        (rate - dividend_yield) * maturity
        + (1.0 - decay) * (theta - v0) / (2.0 * kappa)
        - 0.5 * theta * maturity
    )
    c2 = v0 / (4.0 * kappa**3) * (
        4.0 * kappa**2 * (1.0 + (rho * sigma * maturity - 1.0) * decay)
        + kappa * (4.0 * rho * sigma * (decay - 1.0) - 2.0 * sigma**2 * maturity * decay)
        + sigma**2 * (1.0 - decay**2)
    ) + theta / (8.0 * kappa**3) * (
        8.0 * kappa**3 * maturity
        - 8.0 * kappa**2 * (1.0 + rho * sigma * maturity + (rho * sigma * maturity - 1.0) * decay)
        + 2.0
        * kappa
        * ((1.0 + 2.0 * decay) * sigma**2 * maturity + 8.0 * (1.0 - decay) * rho * sigma)
        + sigma**2 * (decay**2 + 4.0 * decay - 5.0)
    )
    return float(c1), float(c2)


def _validate_contract_inputs(
    spot: float,
    strikes: FloatArray,
    maturity: float,
    rate: float,
    dividend_yield: float,
) -> None:
    if not math.isfinite(spot) or spot <= 0:
        raise ValueError("spot must be positive and finite")
    if strikes.ndim != 1 or strikes.size == 0:
        raise ValueError("strikes must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(strikes)) or np.any(strikes <= 0):
        raise ValueError("strikes must be positive and finite")
    if not math.isfinite(maturity) or maturity < 0:
        raise ValueError("maturity must be finite and non-negative")
    if not math.isfinite(rate) or not math.isfinite(dividend_yield):
        raise ValueError("rate and dividend_yield must be finite")


def _put_coefficients(
    strikes: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    frequencies: FloatArray,
) -> FloatArray:
    interval = upper - lower
    payoff_upper = np.minimum(upper, 0.0)
    active = payoff_upper > lower
    distance = payoff_upper - lower
    angles = distance[:, None] * frequencies[None, :]
    exp_lower = np.exp(lower)[:, None]
    exp_upper = np.exp(payoff_upper)[:, None]
    omega = frequencies[None, :]
    chi = (np.cos(angles) * exp_upper - exp_lower + omega * np.sin(angles) * exp_upper) / (
        1.0 + omega**2
    )
    psi = np.empty_like(chi)
    psi[:, 0] = distance
    psi[:, 1:] = np.sin(angles[:, 1:]) / omega[:, 1:]
    coefficients = 2.0 * strikes[:, None] * (-chi + psi) / interval[:, None]
    coefficients[~active, :] = 0.0
    return coefficients


def cos_prices(
    option_type: str | OptionType,
    spot: float,
    strikes: ArrayLike,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    settings: COSSettings = _DEFAULT_COS_SETTINGS,
) -> FloatArray:
    """Price European options for one maturity with the Heston COS method."""
    kind = OptionType.parse(option_type)
    strike_array = np.asarray(strikes, dtype=float)
    _validate_contract_inputs(spot, strike_array, maturity, rate, dividend_yield)
    if maturity == 0:
        if kind is OptionType.CALL:
            return np.maximum(spot - strike_array, 0.0)
        return np.maximum(strike_array - spot, 0.0)

    c1, c2 = heston_cumulants(maturity, parameters, rate, dividend_yield)
    half_width = max(settings.truncation_width * math.sqrt(abs(c2)), 1e-8)
    log_moneyness = np.log(spot / strike_array)
    center = log_moneyness + c1
    lower = center - half_width
    upper = center + half_width
    frequencies = np.arange(settings.terms, dtype=float) * math.pi / (2.0 * half_width)
    characteristic = heston_log_return_characteristic(
        frequencies, maturity, parameters, rate, dividend_yield
    )
    phase = characteristic[None, :] * np.exp(
        1j * (log_moneyness - lower)[:, None] * frequencies[None, :]
    )
    coefficients = _put_coefficients(strike_array, lower, upper, frequencies)
    terms = np.real(phase * coefficients)
    terms[:, 0] *= 0.5
    put_prices = math.exp(-rate * maturity) * np.sum(terms, axis=1)
    prices = (
        put_prices
        if kind is OptionType.PUT
        else put_prices
        + spot * math.exp(-dividend_yield * maturity)
        - strike_array * math.exp(-rate * maturity)
    )

    discounted_spot = spot * math.exp(-dividend_yield * maturity)
    discounted_strikes = strike_array * math.exp(-rate * maturity)
    if kind is OptionType.CALL:
        lower_bounds = np.maximum(discounted_spot - discounted_strikes, 0.0)
        upper_bounds = np.full_like(strike_array, discounted_spot)
    else:
        lower_bounds = np.maximum(discounted_strikes - discounted_spot, 0.0)
        upper_bounds = discounted_strikes
    tolerances = 1e-8 * np.maximum(1.0, upper_bounds)
    if (
        np.any(~np.isfinite(prices))
        or np.any(prices < lower_bounds - tolerances)
        or np.any(prices > upper_bounds + tolerances)
    ):
        raise ArithmeticError("COS pricing produced a non-finite or out-of-bounds value")
    return np.minimum(np.maximum(prices, lower_bounds), upper_bounds)


def price(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    settings: COSSettings = _DEFAULT_COS_SETTINGS,
) -> float:
    """Price one European call or put with the Heston COS method."""
    return float(
        cos_prices(
            option_type,
            spot,
            [strike],
            maturity,
            parameters,
            rate,
            dividend_yield,
            settings=settings,
        )[0]
    )


def price_with_diagnostics(
    option_type: str | OptionType,
    spot: float,
    strike: float,
    maturity: float,
    parameters: HestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    settings: COSSettings = _DEFAULT_COS_SETTINGS,
) -> HestonPriceResult:
    """Price one option and expose bounds, truncation, and Feller diagnostics."""
    model_price = price(
        option_type,
        spot,
        strike,
        maturity,
        parameters,
        rate,
        dividend_yield,
        settings=settings,
    )
    lower_bound, upper_bound = discounted_price_bounds(
        option_type, spot, strike, maturity, rate, dividend_yield
    )
    if maturity == 0:
        truncation_lower = truncation_upper = math.log(spot / strike)
    else:
        c1, c2 = heston_cumulants(maturity, parameters, rate, dividend_yield)
        half_width = max(settings.truncation_width * math.sqrt(abs(c2)), 1e-8)
        center = math.log(spot / strike) + c1
        truncation_lower = center - half_width
        truncation_upper = center + half_width
    return HestonPriceResult(
        price=model_price,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        terms=settings.terms,
        truncation_lower=truncation_lower,
        truncation_upper=truncation_upper,
        feller_margin=parameters.feller_margin,
        feller_satisfied=parameters.feller_satisfied,
    )
