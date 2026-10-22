# Append-only basket input store

VolForge v1.6 replaces the basket replay's trust in embedded component/correlation JSON with an
optional, fail-closed storage boundary. The dedicated `BasketInputStore` retains canonical
component markets and pairwise correlations as revisioned snapshot packages, archives every raw
artifact by SHA-256, chains ingestion manifests, and resolves only records that were received by a
declared formation time.

The existing CSV replay remains available for controlled experiments. Set
`BasketReplaySettings(require_store_provenance=True)` or pass `--require-store-provenance` when a
workflow must reject unbound CSV rows.

## Canonical record schema

Every CSV or HTTPS adapter artifact uses these columns:

| Column | Meaning |
| --- | --- |
| `record_type` | `component` or `correlation`. |
| `snapshot_id` | Provider package identity shared by every row in one market snapshot. |
| `market_timestamp` | Time the observed market/model input applies. Must include a timezone. |
| `received_timestamp` | Time the source made that record available. Must not precede market time. |
| `source_record_id` | Stable source-local row identity across revisions. |
| `revision` | Non-negative, strictly increasing integer for that source record. |
| `record_action` | `upsert` or `cancel`. |
| `basket_id` | Stable identifier for the modeled basket/package. |
| `asset_id` | Component identity, or the lexicographically first member of a correlation pair. |
| `paired_asset_id` | Second pair member for correlation rows; empty for components. |
| `spot`, `volatility`, `dividend_yield` | Component risk-neutral GBM inputs. |
| `bid`, `ask` | Executable component quote; the spot must lie inside it. |
| `correlation` | Pairwise Brownian correlation in `[-1, 1]`. |

New snapshot packages must contain each component exactly once and every unordered off-diagonal
pair exactly once. Pair identities use lexical order and the diagonal is inferred as one. Snapshot
resolution constructs the full matrix and rejects a non-positive-semidefinite result rather than
repairing it.

## Time and revision semantics

The store keeps three independent time axes:

```text
market_timestamp <= received_timestamp <= batch ingested_timestamp
```

At formation `t`, an as-known query admits only records with market and receipt times at or before
`t`. An optional `dataset_known_at` additionally selects a historical local database version.
Corrections cannot move receipt time backward, change record identity fields, or overwrite prior
rows. Backfills and corrections require a reason; incremental batches cannot silently cross the
source market-time watermark.

`resolve_snapshot` selects the newest eligible market time. It does not fall back from an invalid
newest package to an older convenient one. It requires one unambiguous source/snapshot identity,
complete component and pair coverage, fresh data, executable spreads, and a valid correlation
matrix.

## Persistence and integrity

SQLite triggers reject updates and deletes to batches or records. Each ingestion retains:

- the exact raw CSV or JSON response in a content-addressed sibling directory;
- a canonical normalized payload and SHA-256 for every record;
- row counts, type counts, market-time bounds, adapter metadata, and raw-artifact hash; and
- a manifest hash linked to the preceding batch manifest.

`verify()` recomputes database integrity, record payloads, raw sizes/hashes, counts, time ranges,
and the complete manifest chain. Snapshot resolution and replay materialization refuse an invalid
store.

The materializer adds `input_store_snapshot_sha256` and `input_manifest_hashes_json` to every
replay row. The v1.5 ledger now binds those values into its normalized input hash and audit hash.
This makes later modification evident. A CSV exported from the store still needs the retained
database and raw archives for independent provenance verification.

## Source adapters

Two adapters target the same canonical schema:

- `CsvBasketInputAdapter` reads a bounded local canonical CSV.
- `HttpsJsonBasketInputAdapter` accepts only HTTPS without URL credentials/query secrets, uses the
  platform TLS verifier, limits timeout and response bytes, rejects duplicate JSON fields, and
  accepts a JSON record list or `{ "dataset_id": ..., "records": [...] }`.

The HTTPS adapter can read a bearer token from an environment-variable name supplied by the
operator. It never persists the token in adapter metadata or exception URLs. Provider-specific
symbol mapping, entitlement, pagination, clock guarantees, and authentication policy remain the
responsibility of a concrete integration layered on this canonical adapter contract.

## CLI workflow

Create the deterministic fixture and store:

```bash
volforge generate-basket-inputs data/basket-inputs.csv data/basket-schedule.csv
volforge basket-input-init data/basket-inputs.sqlite
volforge basket-input-ingest data/basket-inputs.sqlite data/basket-inputs.csv \
  --source volforge-synthetic-v1 --mode initial \
  --ingested-at 2026-05-01T00:00:00Z
```

Inspect and materialize a provenance-bound replay input:

```bash
volforge basket-input-query data/basket-inputs.sqlite reports/known-inputs.csv \
  --formation-at 2026-04-10T20:00:00Z \
  --basket-id VOLFORGE-SYNTHETIC-ADJUSTED --source volforge-synthetic-v1
volforge basket-input-materialize data/basket-inputs.sqlite data/basket-schedule.csv \
  reports/basket-replay-input.csv --basket-id VOLFORGE-SYNTHETIC-ADJUSTED \
  --source volforge-synthetic-v1 --maximum-age 15min
volforge basket-input-verify data/basket-inputs.sqlite \
  --report reports/basket-input-integrity.json
```

Fetch a provider that already publishes the canonical JSON contract:

```bash
volforge basket-input-fetch data/basket-inputs.sqlite \
  https://data.example.com/v1/basket-inputs \
  --source provider-name --mode incremental --ingested-at 2026-10-17T14:00:00Z \
  --authorization-env PROVIDER_TOKEN
```

Run the existing replay with its stronger provenance policy:

```bash
volforge basket-replay reports/basket-replay-input.csv data/basket-deliverable.json \
  reports/basket-ledger.csv --settlement-method physical --option-type call --strike 60 \
  --contract-multiplier 100 --currency USD --underlying-id ACME \
  --expiry 2026-04-30T20:00:00Z --require-store-provenance
```

## Interpretation limits

- Stored spot, volatility, yield, correlation, and quotes are observations supplied by the source.
  The v1.7 estimator can replace supplied volatility/correlation with rolling estimates, but does
  not alter stored records; see [`basket-risk-hedging.md`](basket-risk-hedging.md).
- The v1.8 selector compares three estimators only on historical returns known by formation and
  binds its decision separately; see
  [`basket-covariance-validation.md`](basket-covariance-validation.md).
- TLS authenticates the configured server and encrypts transport. It does not establish market
  accuracy, licensing, exchange authority, or a trusted timestamp.
- Hashes and append-only triggers provide local tamper evidence, not immutability against a host
  administrator or a signature by the upstream source.
- The base materialization/replay remains component-delta only. The v1.7 advanced workflow requires
  three explicit instrument definitions and quotes and maintains a separate accounting ledger.
