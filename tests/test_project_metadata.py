import tomllib
from pathlib import Path

import volforge

ROOT = Path(__file__).resolve().parents[1]


def test_public_version_metadata_is_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]

    assert volforge.__version__ == version
    assert f"version: {version}" in (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f"## [{version}]" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"`{version.rsplit('.', 1)[0]}.x`" in (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert project["urls"]["Repository"] == "https://github.com/gunhoo1124-arch/volforge.git"
    assert 'repository-code: "https://github.com/gunhoo1124-arch/volforge"' in (
        ROOT / "CITATION.cff"
    ).read_text(encoding="utf-8")


def test_readme_does_not_publish_a_developer_checkout_path() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "C:\\Users\\" not in readme
