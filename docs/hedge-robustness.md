# Hedge robustness gate

VolForge v0.7.0 adds an auditable layer above the v0.6 single-run hedging laboratory. It asks
whether a hedge remains numerically stable and risk-reducing when random seeds, path budgets,
rebalance frequency, realized dynamics, and costs change.

Passing is deliberately narrower than claiming that a strategy works in markets. The gate covers
configured synthetic Heston scenarios only.

## Pricing dynamics versus realized dynamics

Option marks and hedge ratios always use one risk-neutral parameter vector

```text
Q = (kappa_Q, theta_Q, xi_Q, rho_Q, v0).
```

The path generator independently accepts

```text
P* = (kappa_path, theta_path, xi_path, rho_path, v0)
mu_path = ex-dividend drift in dS/S.
```

The star emphasizes that these are user-specified physical or stressed dynamics, not statistically
estimated parameters. Under a physical annual total-return assumption `mu`, the ex-dividend spot
drift is `mu-q`. Pricing, discounting, financing, and Greeks continue to use the risk-free rate and
the `Q` parameters. This separation prevents a model-misspecification experiment from silently
repricing the target under the same stressed model that generates its P&L.

Risk-neutral and physical variance expectations need not coincide. Carr and Wu define variance
risk premia from the difference between realized and option-implied variance in
[Variance Risk Premiums](https://academic.oup.com/rfs/article-abstract/22/3/1311/1581057).
Bakshi and Kapadia connect volatility risk premia to delta-hedged option gains in
[Delta-Hedged Gains and the Negative Market Volatility Risk
Premium](https://doi.org/10.1093/rfs/16.2.527).

## Five scenarios

Every final refinement budget is evaluated under:

1. `q_matched`: risk-neutral drift and variance parameters, with baseline costs;
2. `physical_drift`: physical expected return but matched variance parameters;
3. `parameter_misspecification`: physical return plus multiplied `kappa`, `theta`, and
   `vol_of_vol`, and an additive `rho` shift;
4. `stressed_costs`: matched risk-neutral paths with multiplied stock and option costs; and
5. `combined_stress`: physical return, stressed variance parameters, and multiplied costs.

The same seed set is reused between scenarios. This common-random-number design reduces noise in
scenario differences. Seeds remain independent of one another.

Model misspecification can materially change realized hedge-error distributions even when models
look similar in a cross-section of option prices; see Branger, Krautheim, Schlag, and Seeger,
[Hedging under model misspecification](https://onlinelibrary.wiley.com/doi/10.1002/fut.20530).

## Joint refinement and pooling

The configured `path_counts` and `rebalancing_steps` form matched, strictly increasing pairs. Each
level is run for every seed, and terminal P&Ls are pooled across those seeds. The final two levels
must stabilize both mean P&L and sample P&L standard deviation.

For mean P&L, the acceptance limit is

```text
absolute tolerance
+ relative tolerance max(|final mean|, 1)
+ 1.96 sqrt(SE_previous^2 + SE_final^2).
```

The volatility limit uses the same deterministic tolerance plus a normal approximation to the
sampling error of each standard deviation, `s/sqrt(2(n-1))`. These allowances prevent the gate
from interpreting ordinary finite-sample variation as deterministic non-convergence. They are
diagnostics, not formal confidence guarantees for non-normal hedge P&L.

## Strategy acceptance

For each hedged strategy, all of the following must hold:

- final mean and P&L volatility changes remain within their refinement limits;
- P&L volatility is reduced versus unhedged by at least the configured fraction in every scenario;
- the largest seed-mean range is no more than the configured multiple of the pooled per-seed
  standard error;
- option-position clipping and ill-conditioned fallback fractions remain below their limits; and
- every P&L attribution reconciles within the configured floating-point tolerance.

The report is downstream eligible when all scenarios reconcile and at least one hedge strategy
passes. It lists eligible strategies rather than asserting that every increasingly complex hedge
must pass. For example, a delta-gamma hedge may be blocked when its hedge option is unstable while
a delta-vega hedge remains acceptable.

## Interpretation limits

- Path parameters are stresses, not calibrated physical-measure estimates.
- The QE simulation and hedge dates are refined together; the gate does not isolate the two error
  sources.
- Scenario coverage is finite and contains no jumps, liquidity regime changes, execution delay,
  discrete contract selection, or recalibration through time.
- The same Heston family generates and prices all scenarios even when its parameter values differ.
- A strategy can lower dispersion while having an unattractive mean after costs.
- Multiple-seed pooling improves stability but does not replace a power analysis or tail-focused
  study.

Bakshi, Cao, and Chen emphasize that pricing, internal consistency, and hedging are distinct model
yardsticks in [Empirical Performance of Alternative Option Pricing
Models](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1997.tb02749.x). VolForge therefore
keeps a robustness pass separate from any future chronological out-of-sample conclusion.
