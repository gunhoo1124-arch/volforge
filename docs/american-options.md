# American-option engines

VolForge v0.5.0 prices vanilla American calls and puts with three deliberately different numerical
methods. Agreement is evidence about an implementation and its configured resolution, not a proof
that any one result is exact.

## CRR benchmark

`american_binomial_price` uses a recombining Cox-Ross-Rubinstein tree. At each exercise date it
compares intrinsic value with the discounted risk-neutral continuation value. The result includes
the analytic Black-Scholes European value, the early-exercise premium, and a discrete exercise
boundary. Increase `BinomialSettings.steps` until both value and boundary stabilize.

## Heston Longstaff-Schwartz

`heston_lsm_price` simulates full spot and variance paths with the same seeded Andersen QE-M
transition used by the European engine. Backward induction regresses discounted realized cash flow
on

```text
1, S/K, (S/K)^2, v/theta, (v/theta)^2, (S/K)(v/theta).
```

Only in-the-money paths enter each regression. The reported standard error is calculated from the
realized discounted policy cash flows and respects antithetic pairing. It is conditional on the
estimated policy: it does not include regression bias, exercise-date discretization, time-step
bias, or parameter uncertainty. The policy estimate should be checked by changing seed, paths,
exercise dates, and basis design. A non-dividend call under non-negative rates is forced to its
European path value because early exercise is dominated in that case.

The method follows Longstaff and Schwartz's original least-squares simulation approach:
[Valuing American Options by Simulation](https://escholarship.org/uc/item/43n1k4jb).

## Heston obstacle PDE

American exercise turns the Heston PDE into a linear complementarity problem:

```text
V >= payoff,
V_tau - L V >= 0,
(V - payoff)(V_tau - L V) = 0.
```

`heston_american_pde_price` reuses the European solver's non-uniform log-price/variance grid and
sparse Heston operator. Every Crank-Nicolson or implicit Rannacher step is followed by an
Ikonen-Toivanen multiplier projection onto the payoff obstacle. The solver prices calls directly;
American put-call parity is not assumed. It returns the corresponding European PDE value, the
early-exercise premium, active-node fraction, and the boundary at the variance node nearest `v0`.

The splitting follows Ikonen and Toivanen's linear-complementarity method, developed for American
options under Heston stochastic volatility: [Operator splitting methods for pricing American
options](https://doi.org/10.1007/s00211-009-0227-5).

## Validation workflow

For a research result:

1. verify the CRR implementation against the constant-volatility benchmark;
2. increase LSM paths and then exercise dates, repeating with several seeds;
3. increase all PDE grid counts and expand both bounded domains;
4. compare the Heston LSM interval with the independently discretized obstacle PDE; and
5. reject the result when premiums, boundaries, or engine spreads do not stabilize.

The included constant-variance-limit test compares the Heston obstacle PDE with CRR. Tests also
enforce the no-early-exercise identity for non-dividend calls and reproduce the standard CRR
American-put value near 2.3193 for `S=K=40`, `T=1`, `r=6%`, and `sigma=20%`.
