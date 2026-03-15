from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = ROOT / "tests"
CONFTST_PATH = TESTS_ROOT / "conftest.py"
LEGACY_ROOTS = (
    "influence",
    "scholar_tracking",
)


def test_root_tests_do_not_reference_legacy_influence_package_roots() -> None:
    ignored = set(re.findall(r"test_[A-Za-z0-9_]+\.py", CONFTST_PATH.read_text(encoding="utf-8")))
    for path in sorted(TESTS_ROOT.glob("test_*.py")):
        if path.name in ignored:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in LEGACY_ROOTS, f"{path.name}: import {alias.name}"
                    assert not any(
                        alias.name.startswith(f"{root}.") for root in LEGACY_ROOTS
                    ), f"{path.name}: import {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in LEGACY_ROOTS, f"{path.name}: from {node.module} import ..."
                assert not any(
                    node.module.startswith(f"{root}.") for root in LEGACY_ROOTS
                ), f"{path.name}: from {node.module} import ..."
