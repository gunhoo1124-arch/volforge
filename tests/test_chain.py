from io import StringIO

import pandas as pd
import pytest

from volforge.data import (
    ChainSchemaError,
    SyntheticChainConfig,
    enrich_option_chain,
    generate_synthetic_chain,
    load_csv,
)
from volforge.diagnostics import put_call_parity_residuals, quality_summary


def test_synthetic_generation_is_deterministic_and_complete() -> None:
    config = SyntheticChainConfig(seed=42, quote_noise_std=0.01)
    first = generate_synthetic_chain(config)
    second = generate_synthetic_chain(config)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 4 * 7 * 2
    assert set(first["option_type"]) == {"call", "put"}
    assert first["expiration"].nunique() == 4


def test_default_synthetic_chain_enriches_and_preserves_parity() -> None:
    enriched = enrich_option_chain(generate_synthetic_chain())
    assert enriched["is_valid"].all()
    assert enriched["iv_converged"].all()
    parity = put_call_parity_residuals(enriched)
    assert parity["parity_residual"].abs().max() < 2e-8


def test_missing_schema_columns_raise() -> None:
    with pytest.raises(ChainSchemaError, match="missing required columns"):
        load_csv(StringIO("strike,bid,ask\n100,1,2\n"))


def test_malformed_and_crossed_quotes_are_retained_with_reasons() -> None:
    raw = generate_synthetic_chain().iloc[:3].copy()
    raw.loc[0, "bid"] = 5.0
    raw.loc[0, "ask"] = 4.0
    raw["strike"] = raw["strike"].astype("object")
    raw.loc[1, "strike"] = "not-a-number"
    raw.loc[2, "bid"] = -1.0
    enriched = enrich_option_chain(raw)
    assert len(enriched) == 3
    assert enriched.loc[0, "flag_crossed_quote"]
    assert "crossed quote" in enriched.loc[0, "validation_reasons"]
    assert "invalid strike" in enriched.loc[1, "validation_reasons"]
    assert enriched.loc[2, "flag_negative_quote"]
    assert not enriched["is_valid"].any()


def test_diagnostic_synthetic_chain_flags_expected_failures() -> None:
    raw = generate_synthetic_chain(SyntheticChainConfig(inject_quality_issues=True))
    enriched = enrich_option_chain(raw)
    summary = quality_summary(enriched)
    assert summary["crossed_quotes"] >= 1
    assert summary["negative_quotes"] >= 1
    assert summary["above_bound_rows"] >= 1
    assert summary["duplicate_rows"] == 2
    assert summary["stale_rows"] >= 1
    assert summary["invalid_rows"] >= 5


def test_act_365f_maturity() -> None:
    raw = generate_synthetic_chain(
        SyntheticChainConfig(expiration_days=(30,), strike_multipliers=(1.0,))
    )
    enriched = enrich_option_chain(raw)
    assert enriched["maturity_years"].iloc[0] == pytest.approx(30.0 / 365.0)
