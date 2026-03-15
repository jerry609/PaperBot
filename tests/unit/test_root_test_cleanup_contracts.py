from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = ROOT / "tests"
LEGACY_ROOTS = (
    "agents",
    "tools",
    "reports",
)
LEGACY_MODULES = (
    "utils.db",
)


def test_tests_do_not_reference_removed_package_roots() -> None:
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in LEGACY_ROOTS, f"{path.relative_to(TESTS_ROOT)}: import {alias.name}"
                    assert alias.name not in LEGACY_MODULES, f"{path.relative_to(TESTS_ROOT)}: import {alias.name}"
                    assert not any(
                        alias.name.startswith(f"{root}.") for root in LEGACY_ROOTS
                    ), f"{path.relative_to(TESTS_ROOT)}: import {alias.name}"
                    assert not any(
                        alias.name.startswith(f"{module}.") for module in LEGACY_MODULES
                    ), f"{path.relative_to(TESTS_ROOT)}: import {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in LEGACY_ROOTS, f"{path.relative_to(TESTS_ROOT)}: from {node.module} import ..."
                assert node.module not in LEGACY_MODULES, f"{path.relative_to(TESTS_ROOT)}: from {node.module} import ..."
                assert not any(
                    node.module.startswith(f"{root}.") for root in LEGACY_ROOTS
                ), f"{path.relative_to(TESTS_ROOT)}: from {node.module} import ..."
                assert not any(
                    node.module.startswith(f"{module}.") for module in LEGACY_MODULES
                ), f"{path.relative_to(TESTS_ROOT)}: from {node.module} import ..."
