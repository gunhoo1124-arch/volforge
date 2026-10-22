# Third-party notices

VolForge's original repository code and documentation are licensed under the MIT License. This
file records known third-party boundaries; it is not a substitute for the license metadata shipped
by each installed package.

## Streamlit theme and interface facilities

The color and layout settings in `.streamlit/config.toml` were adapted from a theme bundled with
[Streamlit](https://github.com/streamlit/streamlit). Streamlit is copyright Streamlit Inc. and
Snowflake Inc. and is distributed under the
[Apache License 2.0](licenses/Apache-2.0.txt). VolForge also refers to Material icon names through
Streamlit's public interface; no icon font or image asset is stored in this repository.

## Python dependencies

VolForge does not vendor dependency source code. Dependencies are installed from the declarations
and hashes in `pyproject.toml` and `uv.lock`, and each dependency remains under its own license. A
release-candidate metadata audit identified these licenses for the direct runtime dependencies:

| Package | Declared license |
| --- | --- |
| Altair | BSD-3-Clause |
| certifi | MPL-2.0 |
| cryptography | Apache-2.0 OR BSD-3-Clause |
| NumPy | BSD-3-Clause and permissive component licenses |
| pandas | BSD-3-Clause |
| Pillow | MIT-CMU |
| SciPy | BSD-3-Clause |
| Streamlit | Apache-2.0 |

The lockfile includes transitive packages under permissive licenses including Apache-2.0,
BSD-family, MIT-family, MPL-2.0, PSF-2.0, CC0-1.0, and Zlib. Review the installed distributions'
license files when redistributing a bundled environment or executable; this repository distributes
neither.

## Research references and data

Documentation links to papers, standards, exchange material, and regulator or industry guidance to
explain financial conventions and numerical methods. Those references remain the property of their
respective authors and publishers. VolForge does not bundle their article text, source code, market
data, logos, or trademarks. Repository fixtures are synthetic and deterministic; production market
and reference data must be supplied by users under their own licenses and entitlements.

If repository material is later copied or adapted from another source, preserve its notices and add
the source, copyright holder, license, affected paths, and modification summary here before release.
