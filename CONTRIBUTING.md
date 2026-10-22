# Contributing to VolForge

## Development setup

```bash
uv sync --locked --dev
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv build
```

Use Python 3.12 through `uv`; do not install project packages globally. Keep the numerical engine,
validation, reporting, CLI, and Streamlit presentation layers separate.

## Change expectations

- Add type hints to every public interface and docstrings to exported numerical behavior.
- Add independent numerical checks for any model formula. Prefer identities, limiting cases,
  published benchmark values, or finite differences over tests that repeat the implementation.
- Preserve structured failure information. Never silently coerce invalid financial inputs or mark
  a failed calculation successful.
- Keep random generation seeded and reports deterministic.
- Keep model benchmarks strictly chronological: score a snapshot before it can enter training,
  preserve the fitted-window timestamps, and bind row-level prediction output into the audit.
- Cluster model-comparison evidence by evaluation date and preserve serial dependence; never
  claim row-level option quotes are independent forecast replications.
- Keep forecast-comparison procedures semantically separate: the block-bootstrap interval is the
  primary conclusion gate, DM/HAC tests equal mean loss, and the date-sign test is an
  independence-based directional sensitivity diagnostic.
- Keep research bundles deterministic and verify artifact, manifest, and root hashes without
  extracting untrusted archive paths.
- Keep signature validity, exact bundle binding, and signer trust as separate states. Never accept
  an embedded self-asserted key as identity, log private-key material, or invent cryptographic
  primitives when a maintained standards-based implementation exists.
- Treat midpoint marks, bid/ask-side fills, spread loss, explicit fees, and contract multipliers as
  different accounting concepts; reconcile each component rather than folding them together.
- Reject deliverables the active pricing engine cannot value rather than approximating an adjusted
  contract as an ordinary single-underlying option.
- Keep `contract_multiplier` separate from deliverable quantities. Settlement code may scale the
  aggregate exercise or cash-settlement difference by the multiplier, but must never rewrite an
  adjusted security or cash component to match it.
- Require an explicit mark for every security component, an authoritative fixed amount for
  cash-in-lieu, and an explicit official level for cash settlement. Do not infer missing values,
  currency conversions, or a multi-asset pre-expiry price from terminal accounting.
- Keep basket pricing assumptions explicit: exact risk-factor coverage, constant GBM parameters,
  a positive-semidefinite correlation matrix, and a seed. Never estimate or repair a missing
  correlation inside the pricing call.
- Report Monte Carlo sampling error separately from model and parameter uncertainty. Keep the
  moment-matched lognormal value labeled as an approximation and retain independent one-asset
  closed-form and price-bound checks.
- Do not admit a standalone basket result to the single-factor Heston replay. The dedicated basket
  replay must receive complete component markets, correlations, and executable hedge quotes by
  each formation time and bind them into its audit.
- Preserve basket self-financing identities: option and component spread, explicit costs, cash
  carry, dividends, settlement, and turnover remain separate. Never book a model price as an
  executable option or component fill.
- Treat basket CSV source labels as declarations, not authentication. A source-backed workflow
  must resolve a complete package from the append-only basket-input store and bind its snapshot and
  manifest hashes into replay. Even then, distinguish retained local provenance from vendor
  identity or a trusted timestamp.
- Keep basket-input market, receipt, and ingestion times separate. Corrections must append strictly
  increasing revisions without moving receipt time backward or changing the stable row/package
  identity. Never repair an incomplete correlation package or a non-PSD matrix silently.
- Keep statistical basket estimation point-in-time: use only synchronized complete observations
  received by formation, retain the exact window and record identities, and make annualization,
  shrinkage, and volatility floors explicit. Never forward-fill a missing component silently.
- Keep covariance comparison strictly chronological: every evaluation return must follow its
  training window, numerical stabilization must remain visible beside the raw eigenvalue, and an
  alternative must pass both stability and declared improvement gates before promotion. Bind the
  full selection decision to its estimate and downstream replay audit.
- Fit covariance regime thresholds only on each fold's earlier training returns and apply common
  labels to every candidate. Keep Gaussian and covariance-consistent Student-t scores distinct;
  require declared regime coverage and stress-loss regret before a robust promotion.
- Define nonlinear basket risks by reproducible bumps and common random numbers. A hedge solve must
  report scaled conditioning, position limits, and residuals; do not hide instability behind a
  pseudoinverse or position clipping.
- Define smile scenarios in forward log-moneyness with explicit coefficient bumps. Recompute each
  instrument's coordinate at every formation, report the full singular-spectrum boundary, and
  reject rank-deficient coverage before solving; never label the local proxy a calibrated surface.
- Treat every hedge option as an explicit contract with executable bid/ask quotes. Keep target,
  hedge-option, and component fills, marks, spreads, fees, carry, dividends, liquidation, and
  settlement separate in the self-financing reconciliation.
- Keep option-quote history independent of component/correlation estimates. Preserve market,
  receipt, and ingestion clocks; never change an instrument's role or contract definition across
  revisions; resolve every requested identity as known at formation; and bind the full target and
  ordered hedge snapshot into replay before claiming executable-quote provenance.
- Source adapters must emit the canonical record contract, bound time/size/resources, keep secrets
  out of URLs, logs, metadata, and archives, and retain the exact response bytes. Add provider-
  specific mapping/entitlement rules in separate adapters rather than weakening store validation.
- Treat historical schemas as versioned public interfaces. New migrations must be append-only,
  preserve earlier manifests, and include received-time, ingested-time, correction, cancellation,
  backfill, and tamper tests.
- Treat reference schemas and effective-time semantics as versioned public interfaces. Test source
  publication cutoffs, stable identities, corporate-action boundaries, holidays, early closes,
  daylight-saving offsets, ambiguity, and missing-reference failure paths.
- Never rewrite a stored market event to repair data. Append a higher revision through an explicit
  correction or add older missing events through a reasoned backfill batch.
- Never infer contract terms or exchange sessions from a symbol when point-in-time reference data
  are required. Add a source-backed reference record and preserve its publication timestamp.
- Do not add proprietary market data, credentials, or paid-API requirements.
- Update `CHANGELOG.md`, relevant docs, and the roadmap when behavior changes.
- Keep Streamlit formulas out of page scripts; pages must call package APIs.

## Pull requests

Explain the financial convention, algorithm, numerical tolerances, test evidence, and any model
limitations. Small, reviewable milestones are preferred. Formatting, lint, tests, and the headless
Streamlit smoke test must pass before review. Complete the pull-request checklist, keep commits free
of generated data and credentials, and confirm that a clean wheel install can import `volforge` and
run `volforge --help`.

## Contribution rights and provenance

Only submit work that you created or have the legal right to contribute. By opening a contribution,
you agree that it is provided under the repository's MIT License and represent that you may grant
those rights. Preserve copyright, license, and attribution notices for permitted third-party
material, identify that material explicitly in the pull request, and update
`THIRD_PARTY_NOTICES.md` when it becomes part of the repository.

Do not copy proprietary code, paywalled publications, vendor datasets, or license-incompatible
snippets. Research citations may explain an algorithm, but implementations must be independently
written and validated. AI-assisted contributions require the same human review, provenance checks,
license review, and responsibility as any other contribution.
