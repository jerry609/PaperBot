from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import ObsidianConfig, create_settings
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore
from paperbot.utils.logging_config import LogFiles, Logger

from .obsidian_exporter import ObsidianFilesystemExporter


def get_obsidian_config() -> ObsidianConfig:
    return create_settings().obsidian


def obsidian_auto_export_enabled(*, for_tracks: bool = False) -> bool:
    config = get_obsidian_config()
    if not config.enabled:
        return False
    if not str(config.vault_path or "").strip():
        return False
    return config.auto_sync_tracks if for_tracks else config.auto_export_on_save


def _list_track_items(
    store: SqlAlchemyResearchStore,
    *,
    user_id: str,
    track_id: int,
    attr_name: str,
) -> list[dict[str, Any]]:
    list_method = getattr(store, attr_name, None)
    if not callable(list_method):
        return []
    items = list_method(user_id=user_id, track_id=track_id, limit=100)
    return list(items) if isinstance(items, list) else []


def export_track_snapshot(
    *,
    user_id: str,
    track_id: int,
    store: Optional[SqlAlchemyResearchStore] = None,
) -> Optional[Dict[str, Any]]:
    config = get_obsidian_config()
    vault_path = Path(config.vault_path).expanduser()
    if not config.enabled or not str(config.vault_path or "").strip():
        return None
    if not vault_path.exists() or not vault_path.is_dir():
        Logger.warning(
            f"Skipping Obsidian export because vault path is unavailable: {vault_path}",
            file=LogFiles.HARVEST,
        )
        return None

    own_store = store is None
    current_store = store or SqlAlchemyResearchStore()
    try:
        track = current_store.get_track(user_id=user_id, track_id=track_id)
        if track is None:
            Logger.warning(
                f"Skipping Obsidian export because track {track_id} was not found for {user_id}",
                file=LogFiles.HARVEST,
            )
            return None

        saved_items = current_store.list_saved_papers(
            user_id=user_id,
            track_id=track_id,
            limit=max(1, int(config.export_limit)),
        )
        track_payload = dict(track)
        track_payload["tasks"] = _list_track_items(
            current_store,
            user_id=user_id,
            track_id=track_id,
            attr_name="list_tasks",
        )
        track_payload["milestones"] = _list_track_items(
            current_store,
            user_id=user_id,
            track_id=track_id,
            attr_name="list_milestones",
        )
        exporter = ObsidianFilesystemExporter()
        template_path = (
            Path(config.paper_template_path).expanduser()
            if config.paper_template_path
            else None
        )
        result = exporter.export_library_snapshot(
            vault_path=vault_path,
            saved_items=saved_items,
            track=track_payload,
            root_dir=config.root_dir,
            paper_template_path=template_path,
            track_moc_filename=getattr(config, "track_moc_filename", "_MOC.md"),
            group_tracks_in_folders=getattr(config, "group_tracks_in_folders", True),
        )
        Logger.info(
            f"Exported track {track_id} snapshot to Obsidian vault {vault_path}",
            file=LogFiles.HARVEST,
        )
        return result
    except Exception as exc:
        Logger.warning(
            f"Obsidian export failed for track {track_id}: {exc}",
            file=LogFiles.HARVEST,
        )
        return None
    finally:
        if own_store and hasattr(current_store, "close"):
            current_store.close()
