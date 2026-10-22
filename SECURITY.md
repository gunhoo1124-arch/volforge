# Security policy

## Supported version

The current `1.11.x` line receives security and data-integrity fixes while it is the latest release.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use **Report a vulnerability** in the
repository's **Security** tab to send reproduction steps, affected versions, and impact through
GitHub private vulnerability reporting. If that option is unavailable, use a private contact
channel published by the repository maintainers and disclose no exploit details publicly. Expect
an acknowledgement within seven days. A disclosure timeline will be coordinated after triage.

## Data and credential handling

VolForge's offline workflows require no network credentials. VolForge includes an optional canonical
HTTPS/JSON basket-input adapter; it reads a bearer token only from an operator-selected environment
variable and does not retain that token in database metadata. Do not commit proprietary chains,
customer data, `.streamlit/secrets.toml`, tokens, SQLite stores, raw archives, or generated reports
that contain sensitive inputs. Uploaded Streamlit CSV content is processed in the active session
and is not intentionally persisted by the application.

The historical, reference, basket-input, and basket-option quote stores are intentionally
persistent. Every ingested artifact is copied byte-for-byte into a content-addressed directory
beside its SQLite database.
Databases and archives are not encrypted automatically; place them in access-controlled storage
appropriate for the data license and sensitivity. Removing the database without its archive, or
the archive without its database, breaks integrity verification.

Reference files may embed licensed symbol directories, adjustment notices, identifiers, or market
calendars. Treat them with the same access, retention, and redistribution controls as quote data.

SHA-256 values and chained manifests detect accidental changes and ordinary tampering. Detached
Ed25519 attestations can authenticate exact bundle bytes to a separately trusted public key, but
they are not trusted timestamps or proof that a vendor supplied the underlying data.

Research-bundle verification is performed in memory, never extracts member paths, rejects
duplicate or undeclared members, and limits both input and declared expanded content to 100 MiB.
The limit is a workstation safeguard, not a substitute for host-level resource isolation when
processing adversarial archives.

Deliverable and security-mark JSON rejects duplicate object fields and non-finite values. CLI and
package string inputs are limited to 1 MiB and deliverables to 64 components. These bounds reduce
accidental resource abuse but are not a hardened parser sandbox; treat externally supplied terms
and marks as untrusted data and verify them against authoritative contract notices.

Basket-market JSON has the same 1 MiB limit and is capped at 16 assets; Monte Carlo requests are
bounded at two million paths. Correlation matrices are checked for finite values, symmetry, unit
diagonal, range, and positive semidefiniteness before allocation and simulation. These checks limit
ordinary workstation misuse but do not make the numerical engine a hostile multi-tenant sandbox.

Basket replay accepts at most 10,000 snapshot rows. Each embedded market and execution document is
bounded, and timing, exact risk-factor coverage, executable quotes, settlement, accounting, and
hash links are checked before a run is eligible. Uploaded CSV source names and record IDs are
untrusted labels: hashes detect later modification to accepted bytes/state but do not authenticate
the supplier, prove market observability, or replace access controls and malware scanning.

Basket-input adapter artifacts are capped at 32 MiB and normalized batches at 100,000 rows. The
HTTPS adapter requires the `https` scheme, the platform TLS verifier, a bounded timeout, and a
bounded response; it rejects URL credentials, query strings, fragments, duplicate JSON fields, and
unexpected response shapes. Do not place tokens in endpoint URLs or command arguments. TLS does
not prove that data are accurate, licensed, exchange-authoritative, or timestamped by a trusted
clock. Provider-specific certificate pinning, entitlement, pagination, symbol mapping, and rate
limits are outside the generic adapter.

Append-only SQLite triggers and chained hashes detect ordinary local modification but do not stop a
host administrator from replacing both a database and its archives. Back up manifest roots through
an independently controlled channel if stronger evidence is required. A provenance-bound replay
CSV commits to store identities but must still be checked against the retained verified store.

Statistical estimate hashes prove that declared settings, observations, manifests, and output
market were bound together; they do not prove that observations are accurate, independently
priced, or suitable for annualization. Advanced hedge JSON is bounded at 1 MiB and requires three
stable instruments in the base mode or five in smile-risk mode, but its bid/ask fields remain
untrusted input. Synthetic fixture quotes are model-derived and are not evidence of executable
liquidity. Treat externally supplied instrument
terms and quotes as market data under the same license, malware-scanning, retention, and access
controls as component inputs.

Basket-option quote artifacts are capped at 20 MiB and normalized batches at 100,000 rows. The
store validates strict contract terms, immutable instrument identities, revision order, quote
sides, and three time axes. Its snapshot hash binds exact target and hedge markets into replay, but
does not authenticate the source label, prove entitlement or quote-size executability, or provide
a trusted timestamp. Keep the database and `.raw` directory together and protect both as licensed
market data.

Smile-risk audit hashes prove that the declared local skew/curvature scenarios, dynamic forward-
log-moneyness coordinates, instrument risks, and rank diagnostics were not later changed. They do
not prove that the scenario is a calibrated volatility surface, the options are independently
executable, or the hedge remains valid outside the local bumps.

Covariance-selection hashes additionally bind chronological training/evaluation folds, losses,
stability diagnostics, and the chosen estimator. They detect later changes and prevent a selection
from being silently attached to a different estimate, but do not establish statistical validity,
normality, independence, or freedom from data poisoning. Restrict write access to source histories
and review estimator promotions through an independently controlled model-change process.

Robust-selection hashes also bind Student-t settings, training-window regime thresholds, realized
energies, calm/stress labels, balanced scores, coverage, and stress-regret decisions. This proves
audit integrity, not that returns follow a Student-t law, regimes are persistent or predictable,
or the input history is representative and free from adversarial contamination.

## Signing-key handling

Private Ed25519 keys and their password files are secrets. Keep them outside the repository, use
encrypted PKCS8 unless an external security boundary protects the key, restrict filesystem access,
back them up separately from the password, and never place either value in command arguments,
logs, Streamlit uploads, or generated reports. VolForge reads passwords from files so they do not
appear directly in ordinary process listings. Delete temporary password files through an approved
secret-management workflow after use.

On POSIX, `research-keygen` applies mode `0600` to the new private-key file. Windows `chmod` cannot
set an equivalent discretionary ACL, so Windows users must restrict the key with NTFS permissions;
the command reports that permissions were not hardened automatically. Key generation and
attestation output refuse to overwrite existing files.

An embedded attestation key proves only that the same key signed the statement. Acceptance requires
an independently obtained `--trusted-public-key`. VolForge does not operate a certificate
authority, timestamp authority, transparency log, revocation service, or key-rotation registry.
Publish fingerprints through an authenticated channel, document rotation dates, and communicate
revocation immediately if a private key or password may have been exposed.

Treat malformed CSV and SQLite files as untrusted input. Create stores with VolForge and ingest
unknown CSVs only under normal workstation controls. Model output is not trading advice and
quality diagnostics do not prove that an arbitrage is executable.
