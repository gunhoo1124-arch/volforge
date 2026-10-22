# Heston numerical engines

VolForge provides two independent ways to value a European option under the same
`HestonParameters` used by the COS reference. Their purpose is numerical triangulation: agreement
across characteristic-function, simulation, and PDE methods is stronger evidence than agreement
within one implementation family.

## Quadratic-exponential Monte Carlo

`heston_monte_carlo` advances conditional variance with Andersen's quadratic-exponential (QE)
approximation. Low conditional variance ratios use a transformed non-central chi-square form;
high ratios use a point mass at zero plus an exponential tail. The log-spot update includes the
QE martingale correction, so its conditional exponential moment is consistent with the
risk-neutral forward.

The reusable `simulate_heston_paths` API also accepts an explicit ex-dividend `spot_drift`. Its
default remains `r-q`; the override exists for v0.7 physical/misspecification experiments and must
not be used for risk-neutral option valuation. Variance parameters are supplied independently by
the caller.

`MonteCarloSettings` exposes path count, time steps, seed, antithetic sampling, the control-variate
switch, and the QE branch threshold. Antithetic mode requires an even number of paths. Gaussian
draws are paired with their negatives and uniforms with `1-U`. The control variate is the
discounted terminal underlying,

```text
X = exp(-rT) S_T,    E[X] = S_0 exp(-qT).
```

The returned standard error treats each antithetic pair as one independent observation. Its 95%
interval describes sampling uncertainty only: it does not include time-step bias, model risk, or
parameter uncertainty. Re-run with more paths and finer time steps before interpreting a mismatch.

The implementation follows Andersen's original
[efficient Heston simulation paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=946405).

## Finite-difference PDE

For log-moneyness `x = log(S/K)` and time-to-maturity `tau`, the European value satisfies

```text
V_tau = 0.5 v V_xx + (r-q-0.5v) V_x
        + rho xi v V_xv + 0.5 xi^2 v V_vv
        + kappa(theta-v) V_v - rV.
```

`heston_pde_price` evolves a bounded put payoff and obtains calls by put-call parity. It uses
non-uniform `x` and variance grids concentrated around the requested spot and low variance,
second-order central differences in the interior, a forward variance derivative at `v=0`, and a
zero-gradient condition at the upper variance boundary. Dirichlet put asymptotics are imposed at
the log-price boundaries.

Time stepping is Crank-Nicolson. The first configured intervals are replaced with two implicit
half-steps (Rannacher damping) to smooth the payoff kink. The sparse left-hand matrices are
factorized once per solve. This full-matrix approach is intentionally transparent and suitable for
moderate research grids; it is not an ADI performance implementation.

The discretization and boundary treatment follow the Heston finite-difference framework discussed
by in 't Hout and Foulon. Their paper also explains why ADI splitting becomes preferable for much
larger grids: [ADI finite-difference schemes for the Heston PDE](https://arxiv.org/abs/0811.3427).

## Automatic convergence gate

`assess_heston_convergence` executes three increasing budgets for each engine. COS raises its term
count and truncation width; Monte Carlo raises paths and time steps while keeping its seed and
variance-reduction policy fixed; the PDE raises all grid counts while expanding both spatial
domains. Every value and configuration is retained as a `ConvergenceObservation`.

The final tolerance is

```text
epsilon = absolute_tolerance
          + relative_tolerance * max(abs(final COS price), 1).
```

COS and PDE estimate their remaining error from the final refinement change. Monte Carlo uses the
larger of its final change and 95% sampling half-width. A report is downstream-eligible only when:

- every engine's estimated error is within its scaled tolerance;
- the Monte Carlo point estimate respects European price bounds;
- the final COS value lies inside the Monte Carlo 95% confidence interval; and
- the spread across the three final engine prices is no larger than `epsilon`.

`require_numerical_convergence` raises `NumericalConvergenceError` for a blocked report. This lets
future calibration, Greek, American-option, or hedging workflows require numerical evidence at an
API boundary rather than relying on UI warnings. The deterministic JSON audit includes the
contract, model parameters, maximum budgets, all nine observations, decisions, and failure reasons.

This is an operational gate, not a proof of asymptotic convergence. Each ladder changes several
resolution controls together, its maximum budget is finite, and the Monte Carlo interval does not
cover parameter or model uncertainty.

## Convergence workflow

For a research result:

1. compute the COS reference with enough terms and truncation width;
2. raise Monte Carlo paths until sampling error is acceptable, then refine time steps;
3. increase log-price, variance, and time grid counts together for the PDE;
4. expand the PDE log-price and variance domains to test truncation sensitivity; and
5. treat persistent disagreement as a diagnostic failure, not as an average to report.

At expiry both engines return intrinsic value. The Monte Carlo result reports whether the point
estimate lies inside European bounds without clipping sampling noise; the PDE rejects material
bound violations. CLI examples are:

```bash
uv run volforge heston-mc --option-type call --spot 100 --strike 100 --maturity 1 \
  --kappa 1.5768 --theta 0.0398 --vol-of-vol 0.5751 --rho -0.5711 \
  --initial-variance 0.0175 --paths 100000 --steps 126 --seed 7

uv run volforge heston-pde --option-type call --spot 100 --strike 100 --maturity 1 \
  --kappa 1.5768 --theta 0.0398 --vol-of-vol 0.5751 --rho -0.5711 \
  --initial-variance 0.0175 --log-price-points 121 --variance-points 61 --time-steps 180

uv run volforge heston-converge --option-type call --spot 100 --strike 100 --maturity 1 \
  --kappa 1.5768 --theta 0.0398 --vol-of-vol 0.5751 --rho -0.5711 \
  --initial-variance 0.0175 --report reports/heston-convergence.json
```

These engines cover European vanilla payoffs. The American exercise layer reuses their QE-M path
transition and Heston spatial operator as described in
[`american-options.md`](american-options.md). Barriers, other path-dependent payoffs, advanced
automatic-differentiation Greeks, and production-scale performance guarantees remain out of scope.
