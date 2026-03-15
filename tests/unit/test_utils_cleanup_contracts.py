from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UTILS_ROOT = ROOT / "src" / "paperbot" / "utils"


def test_orphaned_utils_variants_are_removed() -> None:
    removed = [
        "CCS-DOWN.py",
        "acm_extractor.py",
        "conference_downloader.py",
        "conference_helpers.py",
        "conference_parsers.py",
        "conference_parsers_new.py",
        "downloader - ccs.py",
        "downloader_back.py",
        "downloader_ccs.py",
        "experiment_metrics.py",
        "experiment_runner.py",
        "keyword_optimizer.py",
        "smart_downloader.py",
    ]

    for filename in removed:
        assert not (UTILS_ROOT / filename).exists(), filename


def test_utils_directory_has_no_backup_or_invalid_python_filenames() -> None:
    invalid = []

    for path in UTILS_ROOT.glob("*.py"):
        name = path.name
        if " " in name or name != name.lower():
            invalid.append(name)
        if name.endswith("_back.py") or name.endswith("_new.py"):
            invalid.append(name)

    assert invalid == []
