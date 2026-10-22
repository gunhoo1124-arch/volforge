# Heston Greeks

VolForge v0.5.1 estimates European Heston delta, gamma, vega, theta, and rho with independent
deterministic and simulation methods. The public convention is:

- delta and gamma differentiate with respect to spot;
- vega applies the same absolute shift to `sqrt(v0)` and `sqrt(theta)`;
- theta is calendar decay per year, `-dV/dT`; and
- rho differentiates with respect to the continuous risk-free rate.

The parallel volatility-level convention recovers standard Black-Scholes vega when Heston variance
is nearly constant. It is not a sensitivity to `vol_of_vol`.

The hedging engine deliberately uses a different instantaneous-state vega,
`dV/dsqrt(v_t)` with `theta` fixed. See [`hedging.md`](hedging.md) before comparing the two outputs.

## Deterministic bump-and-revalue

`heston_finite_difference_greeks` supports COS and the European PDE engine. First derivatives use
the five-point central stencil

```text
f'(x) = [f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)] / (12h),
```

and gamma uses the corresponding five-point second derivative. Every result retains the bump and
the discrepancy from a three-point estimate at the same bump. This discrepancy is a stability
diagnostic, not a rigorous truncation bound.

PDE spot bumps reuse one grid centered on the unbumped spot. Values are interpolated cubically in
log-price and linearly in variance. Rebuilding the grid around every bumped spot or using linear
interpolation at the strike can create a numerical kink and a spurious gamma.

## Monte Carlo estimators

For a vanilla payoff, the QE-M terminal spot scales linearly with initial spot. The direct
pathwise estimators are therefore

```text
Delta = exp(-rT) payoff'(S_T) S_T / S_0,
Rho   = sign * T K exp(-rT) 1_ITM,
```

where `sign` is positive for a call and negative for a put.

Conditional on the QE variance path, the independent Gaussian part of terminal log spot has
variance and shock

```text
Q = sum_n q_n,
Z = sum_n sqrt(q_n) epsilon_n,
A = Z / Q.
```

The conditional likelihood-ratio weights are

```text
Delta weight = A / S_0,
Gamma weight = [A^2 - A - 1/Q] / S_0^2.
```

VolForge reports pathwise delta as the primary estimate, likelihood-ratio gamma, and retains the
likelihood-ratio delta as an internal cross-check. Vega and theta use symmetric common-random-number
bumps: each plus/minus simulation reuses the same seed, draws, path count, and time-step count.
Antithetic pairs are averaged before every standard error is calculated.

These direct estimators follow Broadie and Glasserman's pathwise and likelihood-ratio framework:
[Estimating Security Price Derivatives Using Simulation](https://doi.org/10.1287/mnsc.42.2.269).
The conditional terminal-Gaussian construction is related to Giles' hybrid pathwise/LR approach:
[Vibrato Monte Carlo sensitivities](https://ora.ox.ac.uk/objects/uuid%3Adc53ad11-4bd8-480d-9106-7381804c40c2).

## Validation workflow

For a research result:

1. increase COS terms/width or all PDE grid dimensions until deterministic Greeks stabilize;
2. vary each bump and require the five/three-point discrepancy to remain small;
3. increase Monte Carlo paths until the direct-estimator interval is useful;
4. refine Monte Carlo time steps separately because confidence intervals omit discretization bias;
5. repeat with different seeds and check pathwise delta against likelihood-ratio delta; and
6. treat persistent deterministic/Monte Carlo disagreement as a failed diagnostic.

The test suite verifies all five deterministic Greeks against analytic Black-Scholes Greeks in the
nearly constant-variance limit, compares PDE Greeks with COS on a fixed grid, and checks seeded
Monte Carlo estimates against COS with their reported sampling errors.
