# Mathematical conventions

Let `S` be spot, `K` strike, `T` ACT/365F maturity in years, `r` the continuous risk-free rate,
`q` the continuous dividend yield, and `D = exp(-rT)`. Calls use `+`; puts use `-` where a signed
payoff notation is convenient.

## European bounds and parity

The discounted spot is `S exp(-qT)` and discounted strike is `K exp(-rT)`.

- Call bounds: `max(S exp(-qT) - K exp(-rT), 0) <= C <= S exp(-qT)`.
- Put bounds: `max(K exp(-rT) - S exp(-qT), 0) <= P <= K exp(-rT)`.
- Put-call parity: `C - P = S exp(-qT) - K exp(-rT)`.

Bounds are checked before implied-volatility inversion and on every otherwise calculable CSV row.

## Black-Scholes-Merton

For lognormal volatility `sigma > 0`:

```text
d1 = [ln(S/K) + (r - q + sigma^2/2)T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
C  = S exp(-qT) N(d1) - K exp(-rT) N(d2)
P  = K exp(-rT) N(-d2) - S exp(-qT) N(-d1)
```

The implementation uses the standard closed-form delta, gamma, vega, theta, and rho. Theta is
`-dV/dT`, so negative values usually indicate calendar-time decay. Vega is `dV/dsigma` and rho is
`dV/dr`; both are per absolute 1.00 move.

At `T=0`, price is intrinsic value. At `sigma=0`, price is the discounted deterministic forward
payoff. The payoff kink has no classical gamma; VolForge reports a finite gamma of zero and uses a
symmetric exercise indicator at equality.

## Bachelier

VolForge applies the normal model to the forward
`F = S exp((r-q)T)` and discounts the payoff. Normal volatility `sigma_N` has units of price per
square-root year. With `s_N = sigma_N sqrt(T)` and `d = (F-K)/s_N`:

```text
C = D [(F-K) N(d) + s_N n(d)]
P = D [(K-F) N(-d) + s_N n(d)]
```

Unlike lognormal volatility, normal volatility is not divided by spot. Bachelier permits normally
distributed forward changes and remains meaningful near zero or negative forwards, although v0.1
still validates positive input spot and strike for a consistent public contract.

Analytic Greeks account for the dependence of both discount factor and forward on rates. Normal
vega is `D sqrt(T) n(d)` per one price-unit/square-root-year volatility change.

## Implied volatility

Model price is monotone in non-negative volatility. VolForge:

1. validates all contract inputs and the observed price against the relevant bounds;
2. treats a lower-bound price as the explicit zero-volatility solution;
3. expands a non-negative bracket until it contains a root;
4. uses Brent's method by default;
5. optionally attempts Newton updates, accepting only finite updates inside the current bracket;
6. falls back to Brent when vega is too small or Newton leaves the bracket; and
7. reprices the solution and verifies the residual.

An upper-bound price has no finite implied volatility and is reported as a failure. Very deep,
short-dated contracts can have extrinsic value below floating-point resolution; if the model price
is numerically the lower bound, zero volatility is the only identifiable result at the configured
tolerance.

## Volatility-surface coordinates

Surface calibration uses the forward `F = S exp((r-q)T)`, forward log-moneyness
`k = log(K/F)`, and Black-Scholes total implied variance

```text
w(k,T) = sigma_BS(k,T)^2 T.
```

Working in `w` makes cross-expiry monotonicity explicit and avoids comparing annualized implied
volatilities across different maturities. Valid call and put observations for one strike and expiry
are combined by their median implied volatility. Each expiry needs at least five unique strikes and
a surface needs at least two expiries.

## Raw SVI slices

Each raw-SVI slice is

```text
w(k) = a + b [rho (k-m) + sqrt((k-m)^2 + sigma^2)].
```

VolForge fits each expiry with bounded multi-start SLSQP. The calibration constrains `b >= 0`,
`|rho| < 1`, `sigma > 0`, positive minimum total variance, left and right asymptotic slope
magnitudes no greater than 2, and a positive sampled density factor. Per-slice optimization status
alone is not treated as validity. Cross-expiry calendar spreads are checked after all slices fit.

## Power-law SSVI

The joint surface uses the Gatheral-Jacquier SSVI parameterization

```text
w(k,theta) = theta/2 [1 + rho phi(theta) k
                      + sqrt((phi(theta) k + rho)^2 + 1-rho^2)]
phi(theta) = eta theta^(-gamma),  eta > 0,  0 < gamma < 1.
```

`theta` is ATM total variance. VolForge calibrates a nondecreasing `theta` node at each maturity,
one shared `rho`, `eta`, and `gamma`, and enforces at every node

```text
theta phi(theta) (1 + |rho|) < 4
theta phi(theta)^2 (1 + |rho|) <= 4.
```

Together with nondecreasing ATM total variance and the power-law exponent range, these are the
implemented analytic sufficient conditions. VolForge also performs the same sampled diagnostics
used for raw SVI. See the original
[Gatheral-Jacquier SVI paper](https://arxiv.org/abs/1204.0646) for the derivations and scope of the
conditions.

## Static-arbitrage diagnostics and gate

For a twice-differentiable total-variance slice, the sampled butterfly diagnostic is

```text
g(k) = [1 - k w'(k)/(2w(k))]^2
       - w'(k)^2/4 [1/w(k) + 1/4] + w''(k)/2.
```

Positive total variance, `g(k) >= 0`, and admissible wings are the implemented slice checks.
Calendar diagnostics require `w(k,T_far) - w(k,T_near) >= 0` for each adjacent maturity pair on
the configured grid. The downstream gate accepts a surface only when optimization, analytic
conditions where applicable, every butterfly report, and the calendar report pass. Diagnostics on
a finite grid are evidence about the tested domain, not a proof outside it.

## Heston stochastic volatility

Under the risk-neutral measure, VolForge uses

```text
dS_t = (r-q) S_t dt + sqrt(v_t) S_t dW_t^S
dv_t = kappa (theta-v_t) dt + xi sqrt(v_t) dW_t^v
d<W^S,W^v>_t = rho dt.
```

The five calibrated parameters are mean reversion `kappa > 0`, long-run variance `theta > 0`,
volatility of variance `xi > 0`, correlation `-1 < rho < 1`, and initial variance `v0 > 0`.
The Feller margin is `2 kappa theta - xi^2`. A non-negative margin makes zero unattainable for the
variance process; VolForge reports this condition by default and can enforce it through the
calibration parameterization. The pricing model remains defined when the boundary is attainable.

The characteristic function of `X_T = log(S_T/S_0)` is evaluated in little-Heston-trap form:

```text
phi(u) = exp(C(u,T) + D(u,T) v0)
d       = sqrt((kappa-i rho xi u)^2 + xi^2(u^2+iu))
g       = (kappa-i rho xi u-d) / (kappa-i rho xi u+d)

C = iu(r-q)T + (kappa theta/xi^2)
    [(kappa-i rho xi u-d)T - 2 log((1-g exp(-dT))/(1-g))]
D = ((kappa-i rho xi u-d)/xi^2)
    [(1-exp(-dT))/(1-g exp(-dT))].
```

The principal square root is selected with non-negative real part. This affine characteristic
function is the basis of the original
[Heston model](https://doi.org/10.2307/2962057).

## Fourier-cosine pricing

For `y = log(S_T/K)`, the COS method expands the conditional density on `[a,b]` and evaluates

```text
V approximately exp(-rT) sum' Re[phi_y(u_j) exp(-i u_j a)] H_j
u_j = j pi / (b-a).
```

The prime gives the zero-frequency term half weight and `H_j` are analytic payoff coefficients.
VolForge prices the bounded put payoff directly and obtains calls through put-call parity, avoiding
the cancellation sensitivity of the unbounded call payoff. The interval uses corrected analytic
Heston cumulants and a configurable default width of 16 standard deviations. Materially
non-finite or out-of-bound prices raise an error; only floating-point-sized bound deviations are
normalized. The method follows Fang and Oosterlee's
[original COS paper](https://ir.cwi.nl/pub/13283/13283B.pdf).

## Gated Heston calibration

Heston calibration calls `require_downstream_eligible` before reading target variances. It samples
the accepted SVI/SSVI surface in normalized forward coordinates:

```text
F = 1,  K = exp(k),  r = q = 0,
sigma_target(k,T) = sqrt(w(k,T)/T).
```

The optimizer minimizes COS price errors divided by Black-Scholes vega, with a vega floor for
stable wings. This is a first-order implied-volatility residual without running a root solve inside
every objective evaluation. After fitting, every Heston price is independently inverted to obtain
the reported actual implied-volatility RMSE. Optional Feller enforcement parameterizes
`xi = z sqrt(2 kappa theta)` with `0 < z < 1`. Bounds, multiple deterministic starts, convergence,
function evaluations, price residuals, IV residuals, and failures are retained in the result.

## Heston QE Monte Carlo

For a step of length `Delta`, the conditional variance mean and variance are

```text
m  = theta + (v-theta) exp(-kappa Delta)
s2 = v xi^2 exp(-kappa Delta) [1-exp(-kappa Delta)] / kappa
     + theta xi^2 [1-exp(-kappa Delta)]^2 / (2 kappa)
psi = s2 / m^2.
```

The Andersen QE transition matches these moments with a quadratic-Gaussian distribution when
`psi <= 1.5` and a zero-mass/exponential mixture otherwise. VolForge uses the QE-M log-spot update,
which applies the conditional moment-generating function of the chosen variance branch as a
martingale correction. This makes discounted spot the natural control variate with known mean
`S exp(-qT)`. Antithetic Gaussian and uniform pairs are averaged before the standard error is
calculated. A reported confidence interval measures sampling variation, not discretization bias.

## Heston finite-difference PDE

With `x=log(S/K)` and forward time `tau=T-t`, VolForge solves

```text
V_tau = v V_xx/2 + (r-q-v/2) V_x + rho xi v V_xv
        + xi^2 v V_vv/2 + kappa(theta-v) V_v - rV.
```

The put terminal condition is `K max(1-exp(x),0)`. The bounded log-price and variance domain uses
non-uniform sinh grids concentrated around current log-moneyness and low variance. Interior first,
second, and mixed derivatives use central finite differences adapted to unequal spacing. At `v=0`,
all variance-multiplied diffusion terms vanish and `V_v` uses a forward difference. The upper
variance boundary imposes zero `V_v`; log-price boundaries use discounted put asymptotics.

Crank-Nicolson evolves the sparse semi-discrete system. Initial Rannacher intervals replace one
Crank-Nicolson step with two implicit half-steps to damp oscillations from the payoff kink. Calls
are recovered by put-call parity. Grid refinement and domain expansion are required convergence
checks; a finite grid is never presented as the analytic Heston solution.

## Numerical convergence gate

For final COS reference value `C_cos`, the gate uses

```text
epsilon = epsilon_abs + epsilon_rel max(|C_cos|, 1).
```

Each engine is evaluated at three increasing resolution/domain budgets. The final COS and PDE
changes estimate deterministic refinement error. For Monte Carlo the estimate is the maximum of
the final refinement change and `1.9599639845` standard errors. Acceptance requires each engine
estimate and the cross-engine maximum-minus-minimum spread to be no greater than `epsilon`, the
Monte Carlo point estimate to respect European bounds, and the COS reference to fall inside the
Monte Carlo 95% interval. Failure reasons remain separate and are passed unchanged to the hard
downstream gate.

This procedure is intentionally stricter than displaying three prices. Its estimates are empirical
and conditional on the chosen maximum budgets; they are not rigorous truncation bounds or a
substitute for targeted convergence studies in difficult parameter regimes.

## American exercise

The CRR engine applies backward induction on a recombining tree:

```text
V_i = max(payoff(S_i), exp(-r Delta) [p V_up + (1-p) V_down]).
```

Longstaff-Schwartz replaces the unavailable continuation expectation with a least-squares
projection on spot and variance basis functions over in-the-money Heston paths. Its realized
discounted policy cash flows provide a conditional sampling estimate, not a complete error bound.

For the Heston finite-difference engine, American exercise produces the complementarity system

```text
V >= payoff,   V_tau - L V >= 0,
(V-payoff)(V_tau-LV) = 0.
```

Ikonen-Toivanen splitting evolves the sparse PDE predictor and the obstacle multiplier in separate
fractional steps. Calls are solved directly because American put-call parity is not an equality.
See [`american-options.md`](american-options.md) for algorithms, diagnostics, and refinement rules.

## Heston Greek conventions

Deterministic first derivatives use a five-point central stencil; gamma uses its five-point second
derivative counterpart. The same-bump three-point value is retained as an empirical stability
diagnostic. Heston vega is a parallel absolute shift in `sqrt(v0)` and `sqrt(theta)`, theta is
`-dV/dT`, and rho is `dV/dr`.

The QE-M Monte Carlo estimator conditions on the variance path. If `Q` is the accumulated
conditional variance of the independent log-spot Gaussian contribution and `Z` its realized
shock, `A=Z/Q` gives likelihood weights `A/S0` for delta and
`(A^2-A-1/Q)/S0^2` for gamma. Vanilla delta and rho use lower-variance pathwise identities; vega
and theta use paired common-random-number bumps. See [`heston-greeks.md`](heston-greeks.md) for
formulas, estimator scope, and convergence requirements.

## Discrete Heston hedging

For a position short one target option, stock holding `a` and hedge-option holdings `b_j`, the net
local exposures at a hedge date are

```text
Delta_net = a + sum_j b_j Delta_j - Delta_target
Gamma_net =     sum_j b_j Gamma_j - Gamma_target
Vega_net  =     sum_j b_j Vega_j  - Vega_target
Theta_net =     sum_j b_j Theta_j - Theta_target.
```

Delta hedging sets `a=Delta_target`. A one-option gamma or vega hedge sets
`b=X_target/X_hedge`, then resets `a` to remove the hedge option's delta. The two-option strategy
solves the 2-by-2 gamma/vega system and again resets stock delta. VolForge row-scales that system
for its condition diagnostic, falls back to truncated least squares when required, caps option
positions, and calculates the final stock holding after capping.

Over one interval the local attribution is

```text
A_delta = Delta_net Delta S
A_gamma = Gamma_net (Delta S)^2 / 2
A_vega  = Vega_net Delta sqrt(v)
A_theta = Theta_net Delta t.
```

The residual is the exact marked change in stock, hedge options, and target less these terms. Cash
interest, stock dividends, and transaction costs are recorded independently. Because trades are
financed from cash at contemporaneous marks, these components reconcile terminal P&L path by
path. Full conventions and interpretation limits are in [`hedging.md`](hedging.md).

## Hedge robustness statistics

The v0.7 gate distinguishes pricing parameters `Theta_Q` from realized path parameters `Theta_P*`
and permits a separate ex-dividend path drift. Hedge marks and sensitivities use `Theta_Q`; only
the simulated state path uses `Theta_P*` and the stressed drift.

At every joint path/rebalance budget, terminal P&Ls are pooled across independent seeds. For final
and penultimate means `m_f`, `m_p` with standard errors `se_f`, `se_p`, the mean-stability check is

```text
|m_f-m_p| <= eps_abs + eps_rel max(|m_f|,1)
               + 1.9599639845 sqrt(se_f^2+se_p^2).
```

For sample standard deviations, the gate substitutes the approximation
`se(s)=s/sqrt(2(n-1))`. Scenario risk reduction for strategy `h` is

```text
R_h = 1 - standard_deviation(P&L_h) / standard_deviation(P&L_unhedged).
```

Every configured scenario must meet the minimum `R_h`. Seed disagreement is normalized by the
pooled per-seed standard error, and clipping, conditioning fallbacks, and exact P&L reconciliation
are separately bounded. Details are in [`hedge-robustness.md`](hedge-robustness.md).

## Independent checks

Tests compare BSM with canonical published benchmark values, enforce parity and bounds, invert
prices back to input volatility, and compare every analytic Greek with a centered finite
difference. Bachelier is checked against the at-the-money identity
`C = P = sigma_N sqrt(T) / sqrt(2 pi)` when `r=q=0`, plus parity, round trips, and finite
differences.

Surface tests compare analytic derivatives with finite differences, recover synthetic SVI and
SSVI inputs, exercise known negative-density and calendar-crossing cases, validate deterministic
reports, and verify that the downstream gate rejects invalid results.

Heston tests verify characteristic-function identities, the published Fang-Oosterlee benchmark,
the constant-variance Black-Scholes limit, put-call parity, bounds, cumulants, exact synthetic
calibration, optional Feller enforcement, and rejection of a surface that failed the v0.2 gate.
Numerical-engine tests verify seeded Monte Carlo reproducibility, control-variate error reduction,
sampling agreement with COS, PDE refinement toward COS, PDE parity, expiry handling, CLI output,
and headless rendering of the cross-check workspace.
