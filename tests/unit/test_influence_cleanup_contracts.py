from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = ROOT / "tests"
LEGACY_IMPORT_PREFIXES = (
    "from influence.",
    "import influence.",
)


def test_root_tests_do_not_reference_legacy_influence_package_root() -> None:
    for path in sorted(TESTS_ROOT.glob("test_*.py")):
        content = path.read_text(encoding="utf-8")
        for prefix in LEGACY_IMPORT_PREFIXES:
            assert prefix not in content, f"{path.name}: {prefix}"
