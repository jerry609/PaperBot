from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .parser import hash_markdown


@dataclass(frozen=True)
class ObsidianManagedConflict:
    note_path: str
    reason: str
    expected_hash: str
    actual_hash: str
    detected_at: str
    exported_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_managed_conflict(
    *,
    note_path: Path,
    frontmatter: Mapping[str, Any],
    managed_body: str,
) -> Optional[ObsidianManagedConflict]:
    expected_hash = str(frontmatter.get("paperbot_managed_hash") or "").strip()
    if not expected_hash:
        return None

    actual_hash = hash_markdown(managed_body)
    if actual_hash == expected_hash:
        return None

    return ObsidianManagedConflict(
        note_path=str(note_path),
        reason="paperbot_managed_hash_mismatch",
        expected_hash=expected_hash,
        actual_hash=actual_hash,
        detected_at=datetime.now(timezone.utc).isoformat(),
        exported_at=_coerce_optional_string(frontmatter.get("paperbot_exported_at")),
    )


def _coerce_optional_string(value: Any) -> Optional[str]:
    normalized = str(value or "").strip()
    return normalized or None
