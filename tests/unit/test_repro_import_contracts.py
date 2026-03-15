from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = ROOT / "tests"
LEGACY_REPRO_ROOT = "repro"


def test_tests_do_not_reference_legacy_repro_package_root() -> None:
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != LEGACY_REPRO_ROOT, (
                        f"{path.relative_to(TESTS_ROOT)}: import {alias.name}"
                    )
                    assert not alias.name.startswith(f"{LEGACY_REPRO_ROOT}."), (
                        f"{path.relative_to(TESTS_ROOT)}: import {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module != LEGACY_REPRO_ROOT, (
                    f"{path.relative_to(TESTS_ROOT)}: from {node.module} import ..."
                )
                assert not node.module.startswith(f"{LEGACY_REPRO_ROOT}."), (
                    f"{path.relative_to(TESTS_ROOT)}: from {node.module} import ..."
                )
