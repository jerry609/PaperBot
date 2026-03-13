"""VaultExporterPort — export PaperBot artifacts into a note vault."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class VaultExporterPort(Protocol):
    """Abstract interface for exporting PaperBot artifacts into a vault."""

    def export_library_snapshot(
        self,
        *,
        vault_path: Path,
        saved_items: List[Dict[str, Any]],
        track: Optional[Dict[str, Any]] = None,
        root_dir: str = "PaperBot",
        paper_template_path: Optional[Path] = None,
        track_moc_filename: str = "_MOC.md",
        group_tracks_in_folders: bool = True,
    ) -> Dict[str, Any]: ...
