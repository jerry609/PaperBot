from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from config.settings import ObsidianConfig, create_settings
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog
from paperbot.infrastructure.exporters import ObsidianReportExporter
from paperbot.infrastructure.obsidian import (
    ObsidianBidirectionalSync,
    ObsidianVaultWatcher,
    WATCHDOG_AVAILABLE,
)
from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore

router = APIRouter()

_memory_store: Optional[SqlAlchemyMemoryStore] = None
_vault_watcher: Optional[ObsidianVaultWatcher] = None


def _get_memory_store() -> SqlAlchemyMemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = SqlAlchemyMemoryStore()
    return _memory_store


def _get_obsidian_config() -> ObsidianConfig:
    return create_settings().obsidian


def _get_event_log(request: Optional[Request]) -> EventLogPort:
    if request is None:
        return InMemoryEventLog()
    event_log = getattr(request.app.state, "event_log", None)
    return event_log if isinstance(event_log, EventLogPort) else InMemoryEventLog()


def _build_obsidian_sync_service(request: Optional[Request] = None) -> ObsidianBidirectionalSync:
    obsidian_config = _get_obsidian_config()
    vault_value = str(obsidian_config.vault_path or "").strip()
    if not vault_value:
        raise ValueError(
            "vault_path is required. Configure obsidian.vault_path before using Obsidian sync."
        )

    return ObsidianBidirectionalSync(
        vault_path=Path(vault_value),
        root_dir=str(obsidian_config.root_dir or "PaperBot"),
        memory_store=_get_memory_store(),
        event_log=_get_event_log(request),
        sync_state_filename=getattr(
            obsidian_config,
            "sync_state_filename",
            ".paperbot-sync-state.json",
        ),
        pending_dirname=getattr(obsidian_config, "pending_dirname", ".paperbot-pending"),
    )


class ObsidianReportCitationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    year: Optional[int] = None
    authors: List[str] = Field(default_factory=list)
    relevant_finding: str = ""
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    id: Optional[str] = None


class ObsidianReportSectionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = ""
    cited_papers: List[ObsidianReportCitationRequest] = Field(default_factory=list)


class ObsidianMethodComparisonRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    paper: str = Field(..., min_length=1, max_length=500)
    pros: str = ""
    cons: str = ""


class ObsidianExportReportRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    vault_path: Optional[str] = None
    root_dir: Optional[str] = None
    track_name: Optional[str] = Field(default=None, max_length=128)
    workflow_type: str = "research"
    summary: str = ""
    key_insight: str = ""
    sections: List[ObsidianReportSectionRequest] = Field(default_factory=list)
    methods: List[ObsidianMethodComparisonRequest] = Field(default_factory=list)
    trends: str = ""
    future_directions: str = ""
    references: List[ObsidianReportCitationRequest] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class ObsidianExportReportResponse(BaseModel):
    vault_path: str
    root_dir: str
    title: str
    note_path: str


class ObsidianSyncStatusResponse(BaseModel):
    last_synced_at: Optional[str] = None
    pending_count: int
    tracked_note_count: int
    conflict_count: int
    state_path: str
    pending_dir: str
    watchdog_available: bool
    watching: bool


class ObsidianSyncScanResponse(BaseModel):
    last_synced_at: str
    scanned_notes: int
    changed_notes: int
    memories_created: int
    memories_skipped: int
    tag_updates: int
    wikilink_updates: int
    note_updates: int
    conflicts_detected: int
    pending_count: int


class ObsidianWatchResponse(BaseModel):
    watching: bool
    watchdog_available: bool
    mode: str
    root_path: str


@router.post("/obsidian/export-report", response_model=ObsidianExportReportResponse)
def export_obsidian_report(req: ObsidianExportReportRequest) -> ObsidianExportReportResponse:
    obsidian_config = _get_obsidian_config()

    vault_value = str(req.vault_path or obsidian_config.vault_path or "").strip()
    if not vault_value:
        raise HTTPException(
            status_code=400,
            detail="vault_path is required. Pass it in the request or configure obsidian.vault_path.",
        )

    exporter = ObsidianReportExporter()
    try:
        result = exporter.export_report_note(
            vault_path=Path(vault_value),
            report=req.model_dump(),
            root_dir=str(req.root_dir or obsidian_config.root_dir or "PaperBot"),
            track_moc_filename=getattr(obsidian_config, "track_moc_filename", "_MOC.md"),
            group_tracks_in_folders=getattr(obsidian_config, "group_tracks_in_folders", True),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ObsidianExportReportResponse(**result)


@router.get("/obsidian/sync/status", response_model=ObsidianSyncStatusResponse)
def get_obsidian_sync_status(request: Request) -> ObsidianSyncStatusResponse:
    try:
        status = _build_obsidian_sync_service(request).get_status().to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    status["watchdog_available"] = WATCHDOG_AVAILABLE
    status["watching"] = _vault_watcher.is_running if _vault_watcher is not None else False
    return ObsidianSyncStatusResponse(**status)


@router.post("/obsidian/sync/scan", response_model=ObsidianSyncScanResponse)
def scan_obsidian_sync(request: Request) -> ObsidianSyncScanResponse:
    try:
        result = _build_obsidian_sync_service(request).scan().to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ObsidianSyncScanResponse(**result)


@router.post("/obsidian/sync/watch/start", response_model=ObsidianWatchResponse)
def start_obsidian_sync_watch(request: Request) -> ObsidianWatchResponse:
    global _vault_watcher
    obsidian_config = _get_obsidian_config()

    try:
        sync_service = _build_obsidian_sync_service(request)
        sync_service.scan()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if _vault_watcher is not None and _vault_watcher.root_path != sync_service.root_path:
        _vault_watcher.stop()
        _vault_watcher = None

    if _vault_watcher is None:
        _vault_watcher = ObsidianVaultWatcher(
            root_path=sync_service.root_path,
            on_paths_changed=sync_service.sync_paths,
            debounce_seconds=float(getattr(obsidian_config, "sync_debounce_seconds", 1.0)),
        )

    watching = _vault_watcher.start()
    return ObsidianWatchResponse(
        watching=watching,
        watchdog_available=WATCHDOG_AVAILABLE,
        mode="watchdog" if watching else "scan-only",
        root_path=str(sync_service.root_path),
    )


@router.post("/obsidian/sync/watch/stop", response_model=ObsidianWatchResponse)
def stop_obsidian_sync_watch(request: Request) -> ObsidianWatchResponse:
    global _vault_watcher
    try:
        sync_service = _build_obsidian_sync_service(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if _vault_watcher is not None:
        _vault_watcher.stop()
        _vault_watcher = None

    return ObsidianWatchResponse(
        watching=False,
        watchdog_available=WATCHDOG_AVAILABLE,
        mode="scan-only",
        root_path=str(sync_service.root_path),
    )
