from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UTILS_ROOT = ROOT / "src" / "paperbot" / "utils"
SCAN_ROOTS = (
    ROOT / "main.py",
    ROOT / "src",
    ROOT / "tests",
    ROOT / "cli",
)
REMOVED_FILES = (
    "CCS-DOWN.py",
    "acm_extractor.py",
    "conference_downloader.py",
    "conference_helpers.py",
    "conference_parsers.py",
    "conference_parsers_new.py",
    "downloader - ccs.py",
    "downloader_back.py",
    "keyword_optimizer.py",
    "smart_downloader.py",
)
REMOVED_MODULES = (
    "acm_extractor",
    "conference_downloader",
    "conference_helpers",
    "conference_parsers",
    "conference_parsers_new",
    "downloader_back",
    "keyword_optimizer",
    "smart_downloader",
)


def _iter_python_files():
    for path in SCAN_ROOTS:
        if path.is_file():
            yield path
            continue
        yield from sorted(path.rglob("*.py"))


def test_orphaned_utils_variants_are_removed() -> None:
    for filename in REMOVED_FILES:
        assert not (UTILS_ROOT / filename).exists(), filename


def test_repo_does_not_import_removed_utils_modules() -> None:
    removed_imports = {f"paperbot.utils.{name}" for name in REMOVED_MODULES}
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in removed_imports, f"{path.relative_to(ROOT)}: import {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in removed_imports, (
                    f"{path.relative_to(ROOT)}: from {node.module} import ..."
                )


def test_utils_directory_has_no_backup_or_invalid_python_filenames() -> None:
    invalid = []

    for path in UTILS_ROOT.glob("*.py"):
        name = path.name
        if " " in name or name != name.lower():
            invalid.append(name)
        if name.endswith("_back.py") or name.endswith("_new.py"):
            invalid.append(name)

    assert invalid == []
