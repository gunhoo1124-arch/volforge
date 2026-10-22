## Summary

Describe the user or research problem and the change that addresses it.

## Quantitative and data conventions

State relevant formulas, units, clocks, fill/mark conventions, random seeds, tolerances, and
failure behavior. Explain any change to schemas, audit hashes, or downstream gates.

## Validation

- [ ] `uv run ruff format --check .`
- [ ] `uv run ruff check .`
- [ ] `uv run pytest`
- [ ] `uv build` and a clean-wheel import/CLI smoke test
- [ ] Independent benchmark, identity, limiting-case, or finite-difference evidence is included
      for numerical changes

## Review checklist

- [ ] Public interfaces are typed and documented.
- [ ] Randomized behavior is seeded and output remains deterministic where promised.
- [ ] Invalid financial or provenance inputs fail closed with structured diagnostics.
- [ ] Relevant documentation and `CHANGELOG.md` are updated.
- [ ] No credentials, proprietary market data, generated reports, databases, or raw archives are
      included.
- [ ] I created the contribution or have the right to license it under MIT; all third-party
      material and required notices are identified.
- [ ] Backward-compatibility and model limitations are described above.
