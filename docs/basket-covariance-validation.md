# Basket covariance validation

VolForge compares three point-in-time covariance estimators before one is allowed into an adjusted-
basket replay. Selection is chronological: every candidate is fitted to an earlier rolling window
and scores the immediately following return, which is never included in that fit. The selected
estimator is then refitted only to observations known by the requested formation. v1.10 adds an
optional covariance-consistent Student-t score and a regime-balanced selection objective.

This is a forecast-validation gate, not a claim that returns follow the selected scoring law or
that covariance and regimes are stable.

## Candidate estimators

All candidates operate on synchronized component log returns:

- `sample_shrinkage` computes sample volatility and correlation, then shrinks correlation toward
  the identity matrix by `correlation_shrinkage`. This is the v1.7-compatible baseline.
- `ewma` uses normalized exponentially decaying observation weights and an effective-degrees-of-
  freedom correction. `ewma_decay` controls persistence.
- `diagonal_shrinkage` shrinks the sample covariance matrix toward its own diagonal by
  `covariance_shrinkage`, preserving marginal sample variances while reducing cross-covariance.

The estimator name and every parameter are recorded in `BasketEstimationSettings`. No estimator
repairs missing component observations or bypasses the complete-snapshot rule.

## Chronological folds and score

For each of the last `evaluation_returns` historical returns before formation, VolForge:

1. takes the preceding `training_window` returns;
2. fits each candidate without the evaluation return;
3. symmetrizes its covariance and floors eigenvalues only for a numerically defined score;
4. records raw and stabilized minimum eigenvalues plus the stabilized condition number; and
5. scores the next return with both Gaussian and multivariate Student-t negative log-likelihood.

The score for return vector `r`, fitted mean `mu`, dimension `d`, and covariance `Sigma` is:

`0.5 * [d log(2 pi) + log det(Sigma) + (r - mu)' Sigma^-1 (r - mu)]`.

Lower mean held-out NLL is better. This loss emphasizes both forecast scale and realized
Mahalanobis error; a single extreme return can materially affect it. Median NLL is reported as a
diagnostic but never substitutes for the declared mean or regime-balanced selection objective.

For `forecast_distribution=student_t`, the configured degrees of freedom `nu` must exceed two and
`Sigma` remains the forecast covariance rather than an unlabelled scatter matrix. With
`q = (r - mu)' Sigma^-1 (r - mu)`, the covariance-consistent Student-t NLL is:

`lgamma(nu/2) - lgamma((nu+d)/2) + 0.5[d log((nu-2) pi) + log det(Sigma)]`

`+ 0.5(nu+d) log(1 + q/(nu-2))`.

The Gaussian and Student-t losses are both retained per fold; `forecast_distribution` selects the
loss used for comparison.

## Chronological regime diagnostics

Every evaluation return also receives a common calm/stress label that is independent of the
candidate estimator. VolForge fits component medians and median-absolute-deviation scales to that
fold's earlier training window, falling back to sample scale only for a degenerate component. It
computes cross-sectional standardized return energy for the training observations and labels the
next return `stress` when its energy reaches the configured training quantile.

This label uses the realized evaluation return for stratified scoring, but never enters the
preceding covariance fit or threshold. All candidates therefore share the same regime labels. The
`regime_balanced` objective is:

`(1 - stress_weight) * mean_calm_NLL + stress_weight * mean_stress_NLL`.

It blocks when the holdout lacks the configured minimum number of stress folds or at least one
calm fold. `mean_nll` preserves the v1.8 selection behavior and does not require regime coverage.

## Gates and fallback

A candidate is stable only when every fold has finite loss, raw covariance no more negative than
the eigenvalue tolerance, and stabilized condition number no greater than
`maximum_condition_number`. The same eigenvalue and condition checks are repeated on the final
formation-time refit. An unstable candidate cannot be selected.

The lowest-score stable alternative is promoted only if it beats the stable `sample_shrinkage`
baseline by at least `minimum_nll_improvement`. Under `regime_balanced`, it must also keep mean
stress NLL within `maximum_stress_nll_regret` of the baseline. Otherwise VolForge retains the
baseline. Retaining a stable, stress-resilient baseline is an enforced conservative fallback, not
a validation failure. Robust selection additionally blocks on insufficient regime coverage; the
common store, history, point-in-time, PSD, and conditioning gates remain mandatory.

## Provenance contract

`BasketCovarianceSelection` records:

- formation, observation, evaluation, and per-fold training timestamps;
- asset order, both forecast losses, robust energies, common regime labels, candidate calm/stress
  scores, raw eigenvalues, conditions, and gate outcomes;
- best-scoring and actually selected estimators;
- input store snapshot, manifest, and record-payload hashes; and
- a canonical `selection_sha256` self-hash.

`materialize_selected_covariance_replay` writes the full selection JSON and hash beside the
ordinary estimate provenance. `BasketRiskHedgeSettings(require_covariance_selection=True)` then
rejects absent or partial columns, a bad selection self-hash, an ineligible selection, or any
formation, basket, source, or estimator mismatch with the bound estimate. Every accepted
selection hash enters the advanced replay's hash-chained snapshot audit.

## CLI workflow

After creating and ingesting the statistical basket store described in
[`basket-risk-hedging.md`](basket-risk-hedging.md), inspect one formation:

```bash
uv run volforge basket-covariance-select data/statistical-basket.sqlite \
  --basket-id VOLFORGE-SYNTHETIC-ADJUSTED \
  --formation-at 2026-04-01T20:00:00+00:00 \
  --source volforge-synthetic-statistical-v1 \
  --training-window 20 --evaluation-returns 10 \
  --forecast-distribution student_t \
  --student-t-degrees-of-freedom 6 \
  --selection-objective regime_balanced \
  --stress-quantile 0.8 --stress-weight 0.5 \
  --minimum-stress-evaluations 2 --maximum-stress-nll-regret 0 \
  --report reports/covariance-selection.json
```

Materialize the full schedule and require its selection provenance downstream:

```bash
uv run volforge basket-input-select-covariance \
  data/statistical-basket.sqlite data/basket-risk-schedule.csv \
  reports/covariance-selected-replay.csv \
  --basket-id VOLFORGE-SYNTHETIC-ADJUSTED \
  --source volforge-synthetic-statistical-v1 \
  --forecast-distribution student_t \
  --selection-objective regime_balanced
uv run volforge basket-risk-replay \
  reports/covariance-selected-replay.csv data/basket-deliverable.json \
  reports/basket-risk-ledger.csv --report reports/basket-risk-audit.json \
  --settlement-method physical --option-type call --strike 60 \
  --contract-multiplier 100 --currency USD --underlying-id ACME \
  --expiry 2026-04-30T20:00:00+00:00 \
  --maximum-absolute-option-position 150 \
  --require-covariance-selection
```

The larger synthetic position limit is explicit because changing covariance inputs changes the
demo hedge system. Production limits must be independently justified rather than copied from the
fixture.

## Limits

Ten held-out daily returns are far too few for a production model-selection conclusion; the
fixture is intentionally small and deterministic for regression testing. Student-t NLL is a
scoring sensitivity, not an estimated tail law. Calm/stress labels are rolling classifications,
not a transition model, latent-state filter, or forecast of the next regime. VolForge does not yet
provide realized covariance from intraday data, transaction-cost-aware selection, multiple-
comparison adjustment, or live estimator monitoring. Production research should use longer
independently controlled histories, sensitivity analysis across windows, tail parameters and
regime definitions, and explicit change-control review before promoting a model.
