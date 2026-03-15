from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = ROOT / "tests"
LEGACY_REPRO_IMPORT_PREFIXES = (
    "from " + "repro.",
    "import " + "repro.",
)


def test_tests_do_not_reference_legacy_repro_package_root() -> None:
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        content = path.read_text(encoding="utf-8")
        for prefix in LEGACY_REPRO_IMPORT_PREFIXES:
            assert prefix not in content, f"{path.relative_to(TESTS_ROOT)}: {prefix}"
