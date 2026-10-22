# Heston hedging and P&L attribution

VolForge v0.6.0 compares five strategies for a position short one European call or put:

- unhedged;
- delta hedged with stock;
- delta-gamma hedged with stock and the low-strike call;
- delta-vega hedged with stock and the high-strike call; and
- delta-gamma-vega hedged with stock and both calls.

All strategies see the same seeded Andersen QE-M Heston paths. The hedge options have the target
expiry and configurable strikes below and above the target strike. This common-path design makes
differences between strategies attributable to their hedge rules rather than different random
samples.

## Hedge ratios

The target is short one option. If `q_S` denotes stock units and `q_j` hedge-option units, VolForge
chooses long hedge exposures that offset the short target. Delta-only therefore sets

```text
q_S = Delta_target.
```

For a single gamma or vega hedge with sensitivity `X`,

```text
q_H = X_target / X_hedge
q_S = Delta_target - q_H Delta_hedge.
```

The full strategy solves

```text
[Gamma_1 Gamma_2] [q_1] = [Gamma_target]
[ Vega_1  Vega_2] [q_2]   [ Vega_target ]

q_S = Delta_target - q_1 Delta_1 - q_2 Delta_2.
```

Gamma and vega rows are scaled before calculating the condition number, so the diagnostic is not
dominated by their different units. Ill-conditioned systems use a truncated least-squares
fallback. Every hedge-option position is then clipped to the configured cap and stock is reset to
neutralize delta after clipping. The result reports every fallback and clipped position.

## State-dependent risk measures

At each hedge date, the current simulated variance becomes Heston `v0`; `kappa`, `theta`,
`vol_of_vol`, and `rho` stay fixed. COS repricing supplies price and centered three-point finite
differences for delta, gamma, instantaneous vega, and theta. The lightweight three-point stencil
is deliberate because these quantities are evaluated at every path and hedge date; final risk
reports should still use the higher-order estimators in the Heston Greeks workspace.

Hedging vega is

```text
dV / d sqrt(v_t), with theta fixed.
```

This is the sensitivity of the current stochastic state and differs from VolForge's public
parallel-level vega, which shifts both `sqrt(v0)` and `sqrt(theta)`. Theta remains calendar decay
`-dV/dT` per year. A small positive variance floor is used only for COS state repricing when the QE
transition places variance exactly at zero.

## Self-financing ledger and costs

Let `B_i` be cash just before rebalancing, `S_i` spot, `H_ji` hedge-option values, and `Delta q`
the trade into the next holdings. After trading,

```text
B_i+ = B_i - Delta q_S S_i - sum_j Delta q_j H_ji - cost_i.
```

Cash then earns the continuous rate. Stock receives a discrete credit consistent with the input
continuous dividend yield:

```text
B_(i+1)- = exp(r Delta t) B_i+
           + q_S S_i [exp(q Delta t)-1].
```

Stock costs are basis points of traded stock notional. Option costs are basis points of traded
premium. Initial trades, every rebalance, and terminal stock liquidation are included. Expiring
hedge options cash-settle at intrinsic value. Terminal P&L per short target option is

```text
P&L_T = B_T + q_S S_T + sum_j q_j H_jT - target_payoff - liquidation_cost.
```

Transaction costs can make more complete risk neutralization economically inferior. This tension
is a central issue in discrete hedging with costs; Leland's original transaction-cost treatment is
a useful theoretical reference: [Option Pricing and Replication with Transactions
Costs](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1985.tb02383.x).

## Exact attribution

For each holding interval, beginning-of-period net exposures produce local Taylor terms:

```text
P&L_delta = Delta_net Delta S
P&L_gamma = 1/2 Gamma_net (Delta S)^2
P&L_vega  = Vega_net Delta sqrt(v)
P&L_theta = Theta_net Delta t.
```

The interval residual is defined as the exact marked risky-asset change less those four terms.
Financing records cash interest and stock dividends. Transaction cost is a separate negative
component. Consequently,

```text
terminal P&L = delta + gamma + vega + theta
               + financing + transaction cost + residual
```

path by path up to floating-point precision. The residual is not an unexplained accounting gap:
it contains cross-Greeks, higher-order terms, finite-difference error, and moves between discrete
hedge dates. Every strategy reports its maximum absolute reconciliation gap.

## Interpretation and validation

The default v0.6 experiment uses matched risk-neutral dynamics, not historical data. The v0.7
robustness API can instead separate pricing parameters from physical or stressed path dynamics. A
credible study should increase path count, vary seeds, refine the QE and hedge time grids together,
vary COS settings and Greek bumps, inspect hedge-system conditioning, and sweep both cost
assumptions and position caps. Calibrated pricing parameters should come only from a surface that
passed VolForge's arbitrage and numerical gates. See
[`hedge-robustness.md`](hedge-robustness.md) for the multi-scenario gate.

The empirical importance of model choice for option pricing and hedging is examined in
[Empirical Performance of Alternative Option Pricing
Models](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1997.tb02749.x). VolForge does not
claim that a lower in-model P&L dispersion establishes superior out-of-sample performance.
