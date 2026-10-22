"""Model-independent static-arbitrage diagnostics in total-variance space."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ButterflyReport:
    """Sampled positivity and density-factor diagnostics for one smile."""

    minimum_total_variance: float
    minimum_density_factor: float
    variance_violation_count: int
    butterfly_violation_count: int
    left_wing_slope: float
    right_wing_slope: float
    wing_slope_limit: float
    is_arbitrage_free: bool


@dataclass(frozen=True, slots=True)
class CalendarReport:
    """Sampled total-variance monotonicity diagnostics across maturities."""

    minimum_calendar_spread: float
    violation_count: int
    pair_count: int
    is_arbitrage_free: bool


def density_factor(
    log_moneyness: ArrayLike,
    total_variance: ArrayLike,
    first_derivative: ArrayLike,
    second_derivative: ArrayLike,
) -> FloatArray:
    """Return the Gatheral density factor ``g(k)``.

    A strictly positive total-variance slice is free of butterfly arbitrage
    when ``g(k) >= 0`` everywhere and the right-wing call-price limit holds.
    """
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    first = np.asarray(first_derivative, dtype=float)
    second = np.asarray(second_derivative, dtype=float)
    if not (k.shape == w.shape == first.shape == second.shape):
        raise ValueError("density-factor inputs must have identical shapes")
    if np.any(~np.isfinite(k)) or np.any(~np.isfinite(w)):
        raise ValueError("density-factor inputs must be finite")
    if np.any(w <= 0):
        return np.full_like(w, -np.inf, dtype=float)
    return (1.0 - k * first / (2.0 * w)) ** 2 - 0.25 * first**2 * (1.0 / w + 0.25) + 0.5 * second


def butterfly_report(
    log_moneyness: ArrayLike,
    total_variance: ArrayLike,
    first_derivative: ArrayLike,
    second_derivative: ArrayLike,
    *,
    left_wing_slope: float,
    right_wing_slope: float,
    tolerance: float = 1e-8,
    wing_slope_limit: float = 2.0,
) -> ButterflyReport:
    """Summarize sampled butterfly, positivity, and Lee-wing checks."""
    if tolerance < 0 or wing_slope_limit <= 0:
        raise ValueError("tolerance must be non-negative and wing_slope_limit positive")
    w = np.asarray(total_variance, dtype=float)
    g = density_factor(log_moneyness, w, first_derivative, second_derivative)
    variance_violations = int(np.count_nonzero(w < -tolerance))
    butterfly_violations = int(np.count_nonzero(g < -tolerance))
    wings_valid = (
        left_wing_slope >= -tolerance
        and right_wing_slope >= -tolerance
        and left_wing_slope <= wing_slope_limit + tolerance
        and right_wing_slope <= wing_slope_limit + tolerance
    )
    return ButterflyReport(
        minimum_total_variance=float(np.min(w)),
        minimum_density_factor=float(np.min(g)),
        variance_violation_count=variance_violations,
        butterfly_violation_count=butterfly_violations,
        left_wing_slope=float(left_wing_slope),
        right_wing_slope=float(right_wing_slope),
        wing_slope_limit=wing_slope_limit,
        is_arbitrage_free=bool(
            variance_violations == 0 and butterfly_violations == 0 and wings_valid
        ),
    )


def calendar_report(
    maturities: Sequence[float],
    evaluator: Callable[[float, FloatArray], FloatArray],
    log_moneyness: ArrayLike,
    *,
    tolerance: float = 1e-8,
) -> CalendarReport:
    """Check that total variance is nondecreasing across adjacent maturities."""
    maturity_array = np.asarray(maturities, dtype=float)
    k = np.asarray(log_moneyness, dtype=float)
    if maturity_array.ndim != 1 or len(maturity_array) < 2:
        raise ValueError("calendar diagnostics require at least two maturities")
    if np.any(~np.isfinite(maturity_array)) or np.any(np.diff(maturity_array) <= 0):
        raise ValueError("maturities must be finite and strictly increasing")
    if k.ndim != 1 or k.size == 0 or np.any(~np.isfinite(k)):
        raise ValueError("log_moneyness must be a non-empty finite one-dimensional grid")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    minimum = np.inf
    violations = 0
    for near, far in pairwise(maturity_array):
        spread = np.asarray(evaluator(float(far), k), dtype=float) - np.asarray(
            evaluator(float(near), k), dtype=float
        )
        if spread.shape != k.shape or np.any(~np.isfinite(spread)):
            raise ValueError("surface evaluator returned invalid total variances")
        minimum = min(minimum, float(np.min(spread)))
        violations += int(np.count_nonzero(spread < -tolerance))
    return CalendarReport(
        minimum_calendar_spread=float(minimum),
        violation_count=violations,
        pair_count=len(maturity_array) - 1,
        is_arbitrage_free=violations == 0,
    )
