"""Deterministic, presentation-independent report exports."""

from __future__ import annotations

import math

import pandas as pd

from volforge.diagnostics import quality_summary


def dataframe_to_csv(chain: pd.DataFrame) -> bytes:
    """Serialize an enriched chain deterministically without its pandas index."""
    return chain.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")


def quality_report_markdown(chain: pd.DataFrame) -> str:
    """Build a compact deterministic Markdown quality report."""
    metrics = quality_summary(chain)
    mean_residual = metrics["mean_absolute_parity_residual"]
    max_residual = metrics["max_absolute_parity_residual"]

    def number(value: object) -> str:
        return "n/a" if isinstance(value, float) and math.isnan(value) else f"{value:.8g}"

    return "\n".join(
        [
            "# VolForge option-chain quality report",
            "",
            "- Day count: ACT/365F",
            f"- Total rows: {metrics['total_rows']}",
            f"- Valid rows: {metrics['valid_rows']}",
            f"- Invalid rows: {metrics['invalid_rows']}",
            f"- Crossed quotes: {metrics['crossed_quotes']}",
            f"- Bound violations: {metrics['below_bound_rows'] + metrics['above_bound_rows']}",
            f"- Duplicate rows: {metrics['duplicate_rows']}",
            f"- Implied-volatility failures: {metrics['iv_failures']}",
            f"- Put-call parity pairs: {metrics['parity_pairs']}",
            f"- Mean absolute parity residual: {number(mean_residual)}",
            f"- Maximum absolute parity residual: {number(max_residual)}",
            "",
            "Model outputs are research diagnostics, not trading recommendations.",
        ]
    )
