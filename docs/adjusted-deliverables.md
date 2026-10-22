# Adjusted deliverables and settlement accounting

VolForge v1.3 introduced terminal settlement valuation for standard physical, adjusted physical,
and cash-settled vanilla options from explicit contract terms. v1.4 adds a separate pre-expiry
European basket engine under explicit constant correlated-GBM assumptions; it never uses a
one-factor Heston price as a proxy for a multi-security basket.

This distinction follows listed-option mechanics. The Options Industry Council notes that an
adjusted contract can represent something other than the ordinary 100-share unit and that mergers,
splits, and spin-offs can change the deliverable
([adjusted-options glossary](https://prd-web.optionseducation.org/referencelibrary/optionsglossary),
[splits and mergers FAQ](https://www.optionseducation.org/referencelibrary/faq/splits-mergers-spinoffs-bankruptcies)).
OCC adjustment notices can specify a package of shares plus cash in lieu, so VolForge stores the
package rather than reverse-engineering it from an option symbol
([example OCC filing](https://www.theocc.com/getcontentasset/953296e7-b44b-4c1a-8ee7-14c494db771a/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/sr_occ_2025_017.pdf)).

## Component contract

`deliverable_json` is a non-empty JSON list. Each object contains:

| Field | Meaning |
| --- | --- |
| `asset_id` | Stable security, cash, or settlement-value identifier. |
| `asset_type` | `security`, `cash`, `cash_in_lieu`, or `cash_settlement`. |
| `quantity` | Positive units delivered by one option contract. |
| `currency` | Three-letter settlement currency; defaults to the contract currency. |
| `description` | Optional source-backed explanation. |

The settlement engine rejects unknown fields, duplicate JSON keys, duplicate `(asset_type,
asset_id)` pairs, non-finite or non-positive quantities, more than 64 components, and JSON inputs
larger than 1 MiB. Components are sorted before canonical JSON and SHA-256 are generated, so source
row order does not change the deliverable-component identity. Multiplier, settlement method, and
other contract fields remain separate audit fields.

Physical settlement accepts `security`, `cash`, and `cash_in_lieu` components. `cash` and
`cash_in_lieu` quantities are already-resolved currency amounts per contract and therefore have a
unit mark of one. Every security requires an explicit, non-negative unit mark. An unresolved
fractional share is not cash in lieu: it remains a security exposure until the authoritative cash
amount is known.

Cash settlement accepts exactly one `cash_settlement` component whose quantity equals the contract
multiplier. It requires the explicit official settlement level. Cboe describes cash-settled index
options as paying the positive difference between the settlement value and strike, scaled by the
contract multiplier ([cash-settled options](https://www.cboe.com/insights/posts/cash-settled-options-for-options-traders),
[XSP settlement example](https://www.cboe.com/tradable_products/sp_500/mini_spx_options/cash_settlement/)).

VolForge requires every component to use the declared settlement currency. Convert an amount or
mark outside VolForge using a documented FX source before valuation; the engine never invents an
exchange rate.

## Multiplier and quantity

The contract multiplier `M` and physical component quantities are independent terms. For a
physical package with security marks `P_i`, fixed cash amounts `C_j`, and strike `K`:

```text
delivered_value = sum_i(quantity_i * P_i) + sum_j(C_j)
exercise_price  = K * M
call_payoff     = max(delivered_value - exercise_price, 0)
put_payoff      = max(exercise_price - delivered_value, 0)
```

If exercised, a physical call reports the package as the gross receivable and exercise cash as the
gross payable. A physical put reports exercise cash as the gross receivable and the package as the
gross payable. An out-of-the-money contract reports zero exchanged flows and zero payoff.

For a cash-settled contract with official level `S_set`:

```text
call_payoff = M * max(S_set - K, 0)
put_payoff  = M * max(K - S_set, 0)
```

This payoff is the gross cash receivable; there is no delivered security package.

## Package API

```python
from volforge import parse_deliverable_terms, value_option_settlement

terms = parse_deliverable_terms(
    [
        {"asset_id": "ACME", "asset_type": "security", "quantity": 50},
        {"asset_id": "SPIN", "asset_type": "security", "quantity": 10},
        {"asset_id": "USD-CIL", "asset_type": "cash_in_lieu", "quantity": 20},
    ],
    settlement_method="physical",
    contract_multiplier=100,
    currency="USD",
    underlying_id="ACME",
)
valuation = value_option_settlement(
    terms,
    option_type="call",
    strike=60,
    security_marks={"ACME": 120, "SPIN": 20},
)
assert valuation.delivered_value == 6_220
assert valuation.aggregate_exercise_price == 6_000
assert valuation.net_payoff == 220
```

Every supplied security mark must be used and every required security must have a mark. This strict
set equality catches misspelled identifiers instead of silently valuing the wrong package.

## CLI

Store the component list and mark mapping in separate JSON files:

```bash
volforge deliverable-value data/deliverable.json \
  --marks data/security-marks.json --settlement-method physical \
  --option-type call --strike 60 --contract-multiplier 100 \
  --currency USD --underlying-id ACME
```

For cash settlement, omit `--marks` and supply `--settlement-level`:

```bash
volforge deliverable-value data/cash-settlement.json \
  --settlement-method cash --option-type put --strike 600 \
  --settlement-level 595 --contract-multiplier 100 \
  --currency USD --underlying-id SPX
```

Both commands write deterministic JSON to standard output, including canonical deliverable JSON,
its SHA-256, classification, component values, gross flows, net payoff, required marks, and the
single-factor pricing-compatibility flag.

## Replay boundary

Only `standard_physical` terms are compatible with the single-factor SVI/SSVI-Heston replay: one security component,
quantity equal to the multiplier, and identity equal to `underlying_id` when that identifier is
known. `adjusted_physical` and `cash_settled` terms can be valued at settlement, but remain blocked
before surface calibration in that ledger.

That boundary is intentional. Terminal component marks determine an accounting payoff; they do not
provide joint risk-neutral dynamics, cross-asset correlations, dividends, borrow, tradable hedge
instruments, or pre-expiry Greeks. The v1.4 [basket pricing](basket-pricing.md) engine supplies a
static, user-declared correlated-GBM model and pathwise deltas for standalone analysis. It remains
outside the Heston replay because those inputs are not point-in-time surface-calibration records.
The v1.5 [basket hedge replay](basket-replay.md) supplies a separate chronological CSV ledger with
received-by-formation market packages and explicit component execution. VolForge v1.6 can
materialize those rows from the [append-only basket-input store](basket-input-store.md), retaining
revision and manifest evidence. That local evidence still does not authenticate a vendor or
provide a trusted upstream timestamp.
