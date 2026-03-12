from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Request
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

logger = logging.getLogger(__name__)

_MEMORY_STORE_KEY = "obsidian_memory_store"
_VAULT_WATCHER_KEY = "obsidian_vault_watcher"
_RUNTIME_LOCK_KEY = "obsidian_runtime_lock"
_SYNC_LOCK_KEY = "obsidian_sync_lock"

TSyncResult = TypeVar("TSyncResult")


def initialize_obsidian_runtime(app: FastAPI) -> None:
    if getattr(app.state, _RUNTIME_LOCK_KEY, None) is None:
        setattr(app.state, _RUNTIME_LOCK_KEY, threading.Lock())
    if getattr(app.state, _SYNC_LOCK_KEY, None) is None:
        setattr(app.state, _SYNC_LOCK_KEY, threading.Lock())
    if not hasattr(app.state, _MEMORY_STORE_KEY):
        setattr(app.state, _MEMORY_STORE_KEY, None)
    if not hasattr(app.state, _VAULT_WATCHER_KEY):
        setattr(app.state, _VAULT_WATCHER_KEY, None)


def shutdown_obsidian_runtime(app: FastAPI) -> None:
    initialize_obsidian_runtime(app)

    watcher = getattr(app.state, _VAULT_WATCHER_KEY, None)
    stop = getattr(watcher, "stop", None)
    if callable(stop):
        stop()
    setattr(app.state, _VAULT_WATCHER_KEY, None)

    store = getattr(app.state, _MEMORY_STORE_KEY, None)
    close = getattr(store, "close", None)
    if callable(close):
        close()
    setattr(app.state, _MEMORY_STORE_KEY, None)


def _get_runtime_lock(app: FastAPI) -> threading.Lock:
    initialize_obsidian_runtime(app)
    return getattr(app.state, _RUNTIME_LOCK_KEY)


def _get_sync_lock(app: FastAPI) -> threading.Lock:
    initialize_obsidian_runtime(app)
    return getattr(app.state, _SYNC_LOCK_KEY)


def _get_memory_store(app: FastAPI) -> SqlAlchemyMemoryStore:
    initialize_obsidian_runtime(app)
    store = getattr(app.state, _MEMORY_STORE_KEY, None)
    if isinstance(store, SqlAlchemyMemoryStore):
        return store

    with _get_runtime_lock(app):
        store = getattr(app.state, _MEMORY_STORE_KEY, None)
        if isinstance(store, SqlAlchemyMemoryStore):
            return store
        store = SqlAlchemyMemoryStore()
        setattr(app.state, _MEMORY_STORE_KEY, store)
        return store


def _get_vault_watcher(app: FastAPI) -> Optional[ObsidianVaultWatcher]:
    initialize_obsidian_runtime(app)
    watcher = getattr(app.state, _VAULT_WATCHER_KEY, None)
    return watcher if isinstance(watcher, ObsidianVaultWatcher) else None


def _set_vault_watcher(app: FastAPI, watcher: Optional[ObsidianVaultWatcher]) -> None:
    initialize_obsidian_runtime(app)
    setattr(app.state, _VAULT_WATCHER_KEY, watcher)


def _get_obsidian_config() -> ObsidianConfig:
    return create_settings().obsidian


def _get_event_log(app: FastAPI) -> EventLogPort:
    event_log = getattr(app.state, "event_log", None)
    return event_log if isinstance(event_log, EventLogPort) else InMemoryEventLog()


def _build_obsidian_sync_service(app: FastAPI) -> ObsidianBidirectionalSync:
    obsidian_config = _get_obsidian_config()
    vault_value = str(obsidian_config.vault_path or "").strip()
    if not vault_value:
        raise ValueError(
            "vault_path is required. Configure obsidian.vault_path before using Obsidian sync."
        )

    return ObsidianBidirectionalSync(
        vault_path=Path(vault_value),
        root_dir=str(obsidian_config.root_dir or "PaperBot"),
        memory_store=_get_memory_store(app),
        event_log=_get_event_log(app),
        sync_state_filename=obsidian_config.sync_state_filename,
        pending_dirname=obsidian_config.pending_dirname,
    )


def _run_sync_action(
    app: FastAPI,
    action: Callable[[ObsidianBidirectionalSync], TSyncResult],
) -> TSyncResult:
    with _get_sync_lock(app):
        sync_service = _build_obsidian_sync_service(app)
        return action(sync_service)


def _run_scan_task(app: FastAPI) -> None:
    try:
        _run_sync_action(app, lambda sync_service: sync_service.scan())
    except Exception:
        logger.exception("Obsidian background scan failed")


def _run_sync_paths_task(app: FastAPI, paths: List[Path]) -> None:
    try:
        _run_sync_action(app, lambda sync_service: sync_service.sync_paths(paths))
    except Exception:
        logger.exception("Obsidian watcher sync failed")


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
    accepted: bool
    status: str
    mode: str
    root_path: str


class ObsidianWatchResponse(BaseModel):
    watching: bool
    watchdog_available: bool
    mode: str
    root_path: str
    scan_scheduled: bool = False


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
            track_moc_filename=obsidian_config.track_moc_filename,
            group_tracks_in_folders=obsidian_config.group_tracks_in_folders,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ObsidianExportReportResponse(**result)


@router.get("/obsidian/sync/status", response_model=ObsidianSyncStatusResponse)
def get_obsidian_sync_status(request: Request) -> ObsidianSyncStatusResponse:
    try:
        status = _run_sync_action(
            request.app, lambda sync_service: sync_service.get_status()
        ).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    status["watchdog_available"] = WATCHDOG_AVAILABLE
    watcher = _get_vault_watcher(request.app)
    status["watching"] = watcher.is_running if watcher is not None else False
    return ObsidianSyncStatusResponse(**status)


@router.post("/obsidian/sync/scan", response_model=ObsidianSyncScanResponse)
def scan_obsidian_sync(
    request: Request,
    background_tasks: BackgroundTasks,
) -> ObsidianSyncScanResponse:
    try:
        sync_service = _build_obsidian_sync_service(request.app)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    background_tasks.add_task(_run_scan_task, request.app)
    return ObsidianSyncScanResponse(
        accepted=True,
        status="scan_scheduled",
        mode="background",
        root_path=str(sync_service.root_path),
    )


@router.post("/obsidian/sync/watch/start", response_model=ObsidianWatchResponse)
def start_obsidian_sync_watch(
    request: Request,
    background_tasks: BackgroundTasks,
) -> ObsidianWatchResponse:
    obsidian_config = _get_obsidian_config()

    try:
        sync_service = _build_obsidian_sync_service(request.app)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _get_runtime_lock(request.app):
        watcher = _get_vault_watcher(request.app)
        if watcher is not None and watcher.root_path != sync_service.root_path:
            watcher.stop()
            _set_vault_watcher(request.app, None)
            watcher = None

        if watcher is None:
            watcher = ObsidianVaultWatcher(
                root_path=sync_service.root_path,
                on_paths_changed=lambda paths: _run_sync_paths_task(request.app, paths),
                debounce_seconds=obsidian_config.sync_debounce_seconds,
            )
            _set_vault_watcher(request.app, watcher)

        watching = watcher.start()
    background_tasks.add_task(_run_scan_task, request.app)
    return ObsidianWatchResponse(
        watching=watching,
        watchdog_available=WATCHDOG_AVAILABLE,
        mode="watchdog" if watching else "scan-only",
        root_path=str(sync_service.root_path),
        scan_scheduled=True,
    )


@router.post("/obsidian/sync/watch/stop", response_model=ObsidianWatchResponse)
def stop_obsidian_sync_watch(request: Request) -> ObsidianWatchResponse:
    try:
        sync_service = _build_obsidian_sync_service(request.app)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _get_runtime_lock(request.app):
        watcher = _get_vault_watcher(request.app)
        if watcher is not None:
            watcher.stop()
            _set_vault_watcher(request.app, None)

    return ObsidianWatchResponse(
        watching=False,
        watchdog_available=WATCHDOG_AVAILABLE,
        mode="scan-only",
        root_path=str(sync_service.root_path),
    )
