"""
Runbook API Routes - File Management

Provides file management endpoints for DeepCode Studio:
- List/read/write files in project directories
- Snapshot creation and comparison
- File diff and revert functionality
"""

from __future__ import annotations

try:
    import fcntl
except ImportError:
    fcntl = None
    import msvcrt

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from paperbot.application.collaboration.message_schema import new_run_id
from paperbot.infrastructure.stores.models import AgentRunModel, ArtifactModel, Base
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider

router = APIRouter()

_provider: Optional[SessionProvider] = None


def _get_provider() -> SessionProvider:
    global _provider
    if _provider is None:
        _provider = SessionProvider()
        _provider.ensure_tables(Base.metadata)
    return _provider


def _runtime_allowed_dirs_file() -> Path:
    return Path("data/runbook_allowed_dirs.json")


def _acquire_lock(lock_fd) -> None:
    if fcntl is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return

    lock_fd.seek(0)
    lock_fd.write(b"0")
    lock_fd.flush()
    lock_fd.seek(0)
    msvcrt.locking(lock_fd.fileno(), msvcrt.LK_LOCK, 1)


def _release_lock(lock_fd) -> None:
    if fcntl is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        return

    lock_fd.seek(0)
    msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)


def _load_runtime_allowed_dirs() -> List[Path]:
    f = _runtime_allowed_dirs_file()
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [Path(p).resolve() for p in data if isinstance(p, str) and p.strip()]
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return []


def _save_runtime_allowed_dir(dir_path: Path) -> None:
    f = _runtime_allowed_dirs_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    resolved = dir_path.resolve()

    # Use file lock to prevent concurrent read-modify-write races
    lock_path = f.with_suffix(".lock")
    with open(lock_path, "a+b") as lock_fd:
        _acquire_lock(lock_fd)
        try:
            existing = _load_runtime_allowed_dirs()
            if resolved not in existing:
                existing.append(resolved)
            # Atomic write via temp file + rename
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(f.parent), suffix=".tmp", prefix=".allowed_dirs_"
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_f:
                    json.dump(
                        [str(p) for p in existing], tmp_f, ensure_ascii=False, indent=2
                    )
                os.replace(tmp_path, str(f))
            except BaseException:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        finally:
            _release_lock(lock_fd)


def _allowed_workdir_prefixes() -> List[Path]:
    prefixes: List[Path] = [Path(tempfile.gettempdir()).resolve()]
    try:
        prefixes.append(Path.cwd().resolve())
    except Exception:
        pass
    try:
        home_dir = Path.home().resolve()
        prefixes.append((home_dir / "Documents").resolve(strict=False))
    except Exception:
        pass

    extra = os.getenv("PAPERBOT_RUNBOOK_ALLOW_DIR_PREFIXES", "").strip()
    if extra:
        for p in extra.split(","):
            p = p.strip()
            if p:
                prefixes.append(Path(p).expanduser().resolve())

    prefixes.extend(_load_runtime_allowed_dirs())

    # Preserve order and deduplicate
    unique: List[Path] = []
    seen = set()
    for prefix in prefixes:
        key = str(prefix)
        if key in seen:
            continue
        seen.add(key)
        unique.append(prefix)
    return unique


def _is_under_prefix(path: Path, prefix: Path) -> bool:
    try:
        path_str = os.path.normpath(str(path))
        prefix_str = os.path.normpath(str(prefix))
        return path_str == prefix_str or path_str.startswith(prefix_str + os.sep)
    except Exception:
        return False


def _allowed_workdir(workdir: Path) -> bool:
    allowed_prefixes = _allowed_workdir_prefixes()

    try:
        resolved = Path(os.path.normpath(str(workdir)))
    except Exception:
        return False

    for prefix in allowed_prefixes:
        if _is_under_prefix(resolved, prefix):
            return True

    return False


def _normalize_path_input(raw_path: str, field_name: str) -> str:
    raw = raw_path.strip()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty")
    if "\x00" in raw:
        raise HTTPException(status_code=400, detail=f"{field_name} contains invalid character")

    # Manual '~' expansion keeps home shorthand support while avoiding direct
    # path construction from the untrusted input before policy checks.
    if raw == "~":
        normalized = str(Path.home())
    elif raw.startswith("~/"):
        normalized = os.path.join(str(Path.home()), raw[2:])
    else:
        normalized = raw

    if not os.path.isabs(normalized):
        normalized = os.path.join(os.getcwd(), normalized)
    return os.path.abspath(normalized)


def _resolve_real_path_within_prefixes(
    normalized_real: str,
    prefixes: List[Path],
    *,
    field_name: str,
    status_code: int = 403,
    detail: Optional[str] = None,
) -> Path:
    for prefix in prefixes:
        prefix_real = os.path.realpath(str(prefix))
        if normalized_real == prefix_real:
            return prefix
        if normalized_real.startswith(prefix_real + os.sep):
            suffix = normalized_real[len(prefix_real):].lstrip("/\\")
            candidate = (prefix / suffix).resolve(strict=False) if suffix else prefix
            if _is_under_prefix(candidate, prefix):
                return candidate
            break

    raise HTTPException(
        status_code=status_code,
        detail=detail or f"{field_name} is not allowed",
    )


def _normalize_user_directory(raw_path: str, field_name: str) -> Path:
    """
    Normalize a user-supplied directory path and enforce allow-prefix policy.
    """
    normalized_real = _normalize_path_input(raw_path, field_name)
    return _resolve_real_path_within_prefixes(
        normalized_real,
        _allowed_workdir_prefixes(),
        field_name=field_name,
    )


def _allowlist_mutation_roots() -> List[Path]:
    roots: List[Path] = [Path(tempfile.gettempdir()).resolve()]
    try:
        roots.append(Path.cwd().resolve())
    except Exception:
        pass
    try:
        roots.append(Path.home().resolve())
    except Exception:
        pass

    unique: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


class PrepareProjectDirRequest(BaseModel):
    project_dir: str
    create_if_missing: bool = True


@router.post("/runbook/project-dir/prepare")
async def prepare_project_dir(body: PrepareProjectDirRequest):
    """
    Validate and normalize a project directory for runbook operations.

    If the path is under allowed prefixes and does not exist yet, this endpoint
    can create it to make Studio directory selection smoother.
    """
    try:
        resolved = _normalize_user_directory(body.project_dir, field_name="project_dir")
    except HTTPException:
        raise

    created = False
    if resolved.exists() and not resolved.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be a directory")
    if not resolved.exists():
        if not body.create_if_missing:
            raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
        try:
            resolved.mkdir(parents=True, exist_ok=True)
            created = True
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"failed to create project_dir: {exc}") from exc

    return {
        "project_dir": str(resolved),
        "created": created,
        "allowed_prefixes": [str(p) for p in _allowed_workdir_prefixes()],
    }


class AddAllowedDirRequest(BaseModel):
    directory: str


def _runtime_allowlist_mutation_enabled() -> bool:
    return os.getenv("PAPERBOT_RUNBOOK_ALLOWLIST_MUTATION", "false").lower() == "true"


# Directories that must never be added to the allowlist (too broad / sensitive).
_DENIED_PATHS = frozenset({
    "/", "/bin", "/sbin", "/usr", "/etc", "/var", "/root", "/sys", "/proc",
    "/dev", "/boot", "/lib", "/lib64", "/opt",
})


@router.post("/runbook/allowed-dirs")
async def add_allowed_dir(body: AddAllowedDirRequest):
    """Add a directory to the runtime-allowed prefixes list."""
    if not _runtime_allowlist_mutation_enabled():
        raise HTTPException(
            status_code=403,
            detail="runtime allowlist mutation is disabled"
        )

    raw = body.directory.strip()
    if not raw or "\x00" in raw:
        raise HTTPException(status_code=400, detail="invalid directory path")

    normalized_real = _normalize_path_input(raw, field_name="directory")
    resolved = _resolve_real_path_within_prefixes(
        normalized_real,
        _allowlist_mutation_roots(),
        field_name="directory",
        detail="directory is outside the allowed mutation roots",
    )

    denied_roots = {Path(denied).resolve() for denied in _DENIED_PATHS}
    if resolved in denied_roots:
        raise HTTPException(
            status_code=403,
            detail=f"adding '{resolved}' is not allowed — path is too broad or sensitive",
        )

    # Deny home directory itself — too broad
    if resolved == Path.home().resolve():
        raise HTTPException(
            status_code=403,
            detail="adding home directory is not allowed — too broad",
        )

    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(status_code=400, detail="directory does not exist or is not a directory")

    _save_runtime_allowed_dir(resolved)
    return {
        "ok": True,
        "directory": str(resolved),
        "allowed_prefixes": [str(p) for p in _allowed_workdir_prefixes()],
    }


@router.get("/runbook/allowed-dirs")
async def get_allowed_dirs():
    """Return all currently allowed workspace directory prefixes."""
    return {"prefixes": [str(p) for p in _allowed_workdir_prefixes()]}


def _resolve_under_root(root: Path, relative_path: str) -> Path:
    """
    Resolve a relative path within a root directory.
    Rejects path traversal.
    """
    target = (root / relative_path).resolve()
    root_resolved = root.resolve()
    if not (target == root_resolved or str(target).startswith(str(root_resolved) + os.sep)):
        raise HTTPException(status_code=400, detail="invalid path")
    return target


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot Management
# ──────────────────────────────────────────────────────────────────────────────


def _snapshot_root() -> Path:
    root = Path(os.getenv("PAPERBOT_RUNBOOK_SNAPSHOT_DIR", "data/runbook_snapshots"))
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


class CreateSnapshotRequest(BaseModel):
    project_dir: str
    label: str = ""
    max_total_bytes: int = Field(5_000_000, ge=100_000, le=50_000_000)
    max_file_bytes: int = Field(200_000, ge=1_000, le=10_000_000)


@router.post("/runbook/snapshots")
async def create_snapshot(body: CreateSnapshotRequest):
    """
    Create a text snapshot of a project directory for diff/revert.

    Snapshot is stored on disk (data/runbook_snapshots) and indexed as an ArtifactModel (type=snapshot).
    """
    root = Path(body.project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    ignore_dirs = {".git", ".next", "node_modules", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
    allowed_ext = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".sh", ".ini", ".cfg"}

    files: Dict[str, Dict[str, Any]] = {}
    skipped_large: List[str] = []
    skipped_binary: List[str] = []
    total_bytes = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for name in filenames:
            p = Path(dirpath) / name
            rel = os.path.relpath(str(p), str(root))
            ext = p.suffix.lower()
            if ext and ext not in allowed_ext:
                continue
            try:
                size = p.stat().st_size
            except Exception:
                continue
            if size > body.max_file_bytes:
                skipped_large.append(rel)
                continue
            if total_bytes + size > body.max_total_bytes:
                break
            try:
                content = p.read_text(encoding="utf-8")
            except Exception:
                try:
                    content = p.read_text(errors="ignore")
                except Exception:
                    skipped_binary.append(rel)
                    continue
            total_bytes += len(content.encode("utf-8", errors="ignore"))
            files[rel] = {"sha256": _sha256_text(content), "size_bytes": size, "content": content}
        if total_bytes >= body.max_total_bytes:
            break

    created_at = datetime.now(timezone.utc)
    snapshot_payload = {
        "version": 1,
        "project_dir": str(root.resolve()),
        "label": body.label,
        "created_at": created_at.isoformat(),
        "files": files,
        "skipped": {
            "too_large": sorted(skipped_large),
            "binary_or_unreadable": sorted(skipped_binary),
        },
        "limits": {
            "max_total_bytes": body.max_total_bytes,
            "max_file_bytes": body.max_file_bytes,
        },
    }

    run_id = new_run_id()
    with _get_provider().session() as session:
        run = AgentRunModel(
            run_id=run_id,
            workflow="runbook",
            started_at=created_at,
            ended_at=created_at,
            status="completed",
            executor_type=None,
            timeout_seconds=None,
            paper_url=None,
            paper_id=None,
            metadata_json=json.dumps({"kind": "snapshot", "project_dir": str(root), "label": body.label}, ensure_ascii=False),
        )
        session.merge(run)
        session.commit()

        snapshot_path = _snapshot_root() / f"{run_id}.json"
        snapshot_path.write_text(json.dumps(snapshot_payload, ensure_ascii=False), encoding="utf-8")

        artifact = ArtifactModel(
            run_id=run_id,
            step_id=None,
            type="snapshot",
            path_or_uri=str(snapshot_path),
            mime="application/json",
            size_bytes=snapshot_path.stat().st_size if snapshot_path.exists() else None,
            sha256=None,
            created_at=created_at,
            metadata_json=json.dumps(
                {"project_dir": str(root.resolve()), "label": body.label, "file_count": len(files), "total_bytes": total_bytes},
                ensure_ascii=False,
            ),
        )
        session.add(artifact)
        session.commit()

        return {
            "snapshot_id": artifact.id,
            "run_id": run_id,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "skipped": snapshot_payload["skipped"],
        }


def _load_snapshot(snapshot_id: int) -> Dict[str, Any]:
    with _get_provider().session() as session:
        artifact = session.get(ArtifactModel, snapshot_id)
        if artifact is None or artifact.type != "snapshot":
            raise HTTPException(status_code=404, detail="snapshot not found")
        path = Path(artifact.path_or_uri)
        snap_root = _snapshot_root()
        try:
            resolved = path.resolve()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid snapshot path")
        if not (resolved == snap_root or str(resolved).startswith(str(snap_root) + os.sep)):
            raise HTTPException(status_code=400, detail="snapshot path not allowed")
        if not resolved.exists():
            raise HTTPException(status_code=404, detail="snapshot file missing")
        try:
            return json.loads(resolved.read_text(encoding="utf-8"))
        except Exception:
            raise HTTPException(status_code=500, detail="failed to read snapshot")


@router.get("/runbook/snapshots/{snapshot_id}")
async def get_snapshot(snapshot_id: int):
    payload = _load_snapshot(snapshot_id)
    return {
        "snapshot_id": snapshot_id,
        "project_dir": payload.get("project_dir"),
        "label": payload.get("label"),
        "created_at": payload.get("created_at"),
        "file_count": len((payload.get("files") or {}).keys()),
        "files": sorted((payload.get("files") or {}).keys()),
        "skipped": payload.get("skipped") or {},
    }


@router.get("/runbook/diff")
async def diff_file(
    snapshot_id: int = Query(...),
    project_dir: str = Query(...),
    path: str = Query(..., description="Relative file path within project_dir"),
):
    root = Path(project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    snapshot = _load_snapshot(snapshot_id)
    files = snapshot.get("files") or {}
    if path not in files:
        raise HTTPException(status_code=404, detail="file not found in snapshot")
    old_content = files[path].get("content", "")

    target = _resolve_under_root(root, path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found on disk")

    try:
        new_content = target.read_text(encoding="utf-8")
    except Exception:
        new_content = target.read_text(errors="ignore")

    return {
        "snapshot_id": snapshot_id,
        "path": path,
        "old": old_content,
        "new": new_content,
    }


class RevertFileRequest(BaseModel):
    snapshot_id: int
    project_dir: str
    path: str


@router.post("/runbook/revert")
async def revert_file(body: RevertFileRequest):
    root = Path(body.project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    snapshot = _load_snapshot(body.snapshot_id)
    files = snapshot.get("files") or {}
    if body.path not in files:
        raise HTTPException(status_code=404, detail="file not found in snapshot")
    old_content = files[body.path].get("content", "")

    target = _resolve_under_root(root, body.path)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(old_content, encoding="utf-8")
    return {"ok": True, "path": body.path}


# ──────────────────────────────────────────────────────────────────────────────
# File Operations
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/runbook/files")
async def list_project_files(
    project_dir: str = Query(..., description="Project directory on the API host"),
    recursive: bool = Query(True, description="List files recursively"),
    max_files: int = Query(2000, ge=1, le=20000),
):
    """
    List files under a project directory (best-effort).

    Notes:
    - This endpoint is intentionally restrictive and will only serve allowed roots.
    - Large directories (e.g. node_modules) are skipped by default.
    """
    root = Path(project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    ignore_dirs = {".git", ".next", "node_modules", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}

    files: List[str] = []
    directories: List[str] = []

    if not recursive:
        for p in root.iterdir():
            if p.is_dir():
                directories.append(p.name)
            elif p.is_file():
                files.append(p.name)
        return {"project_dir": str(root), "files": sorted(files), "directories": sorted(directories)}

    for dirpath, dirnames, filenames in os.walk(root):
        # prune
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir != ".":
            directories.append(rel_dir)

        for name in filenames:
            if len(files) >= max_files:
                break
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            files.append(rel)
        if len(files) >= max_files:
            break

    return {
        "project_dir": str(root),
        "files": sorted(files),
        "directories": sorted(set(directories)),
        "truncated": len(files) >= max_files,
        "max_files": max_files,
    }


class ReadFileResponse(BaseModel):
    path: str
    content: str


@router.get("/runbook/file", response_model=ReadFileResponse)
async def read_project_file(
    project_dir: str = Query(..., description="Project directory on the API host"),
    path: str = Query(..., description="Relative file path within project_dir"),
    max_bytes: int = Query(2_000_000, ge=1, le=20_000_000),
):
    """Read a single file under project_dir (UTF-8 best effort)."""
    root = Path(project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    target = _resolve_under_root(root, path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    size = target.stat().st_size
    if size > max_bytes:
        raise HTTPException(status_code=413, detail=f"file too large ({size} bytes)")

    try:
        content = target.read_text(encoding="utf-8")
    except Exception:
        content = target.read_text(errors="ignore")

    return ReadFileResponse(path=path, content=content)


class WriteFileRequest(BaseModel):
    project_dir: str
    path: str
    content: str


@router.post("/runbook/file")
async def write_project_file(body: WriteFileRequest):
    """Write a file under project_dir (creates parents)."""
    root = Path(body.project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    target = _resolve_under_root(root, body.path)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body.content, encoding="utf-8")

    return {"ok": True, "path": body.path}


@router.get("/runbook/changes")
async def list_changes(
    snapshot_id: int = Query(...),
    project_dir: str = Query(...),
    max_files: int = Query(5000, ge=1, le=20000),
):
    """
    Compute file-level changes between a snapshot and the current project directory.

    Returns lists of changed/unchanged/added/removed files.
    """
    root = Path(project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    snapshot = _load_snapshot(snapshot_id)
    snap_files = snapshot.get("files") or {}
    snap_paths = set(snap_files.keys())

    ignore_dirs = {".git", ".next", "node_modules", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
    allowed_ext = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".sh", ".ini", ".cfg"}

    changed: List[str] = []
    unchanged: List[str] = []
    removed: List[str] = []

    for rel in sorted(snap_paths):
        target = _resolve_under_root(root, rel)
        if not target.exists() or not target.is_file():
            removed.append(rel)
            continue
        try:
            content = target.read_text(encoding="utf-8")
        except Exception:
            content = target.read_text(errors="ignore")
        current_sha = _sha256_text(content)
        if current_sha != snap_files.get(rel, {}).get("sha256"):
            changed.append(rel)
        else:
            unchanged.append(rel)

    current_files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for name in filenames:
            if len(current_files) >= max_files:
                break
            p = Path(dirpath) / name
            ext = p.suffix.lower()
            if ext and ext not in allowed_ext:
                continue
            rel = os.path.relpath(str(p), str(root))
            current_files.append(rel)
        if len(current_files) >= max_files:
            break

    added = sorted([p for p in current_files if p not in snap_paths])

    return {
        "snapshot_id": snapshot_id,
        "project_dir": str(root.resolve()),
        "changed": changed,
        "unchanged": unchanged,
        "added": added,
        "removed": removed,
    }


class DeleteFileRequest(BaseModel):
    project_dir: str
    path: str


@router.post("/runbook/delete")
async def delete_file(body: DeleteFileRequest):
    """Delete a single file under project_dir."""
    root = Path(body.project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    target = _resolve_under_root(root, body.path)
    if not target.exists():
        return {"ok": True, "path": body.path, "deleted": False}
    if target.is_dir():
        raise HTTPException(status_code=400, detail="cannot delete a directory")

    target.unlink()
    return {"ok": True, "path": body.path, "deleted": True}


class RevertProjectRequest(BaseModel):
    snapshot_id: int
    project_dir: str
    delete_added: bool = True


@router.post("/runbook/revert-project")
async def revert_project(body: RevertProjectRequest):
    """Revert project files back to a snapshot (file-level)."""
    root = Path(body.project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    snapshot = _load_snapshot(body.snapshot_id)
    snap_files = snapshot.get("files") or {}
    snap_paths = set(snap_files.keys())

    restored = 0
    for rel, info in snap_files.items():
        target = _resolve_under_root(root, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(info.get("content", ""), encoding="utf-8")
        restored += 1

    deleted = 0
    if body.delete_added:
        ignore_dirs = {".git", ".next", "node_modules", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
        allowed_ext = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".sh", ".ini", ".cfg"}
        current_files: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
            for name in filenames:
                p = Path(dirpath) / name
                ext = p.suffix.lower()
                if ext and ext not in allowed_ext:
                    continue
                rel = os.path.relpath(str(p), str(root))
                current_files.append(rel)
        for rel in current_files:
            if rel in snap_paths:
                continue
            try:
                target = _resolve_under_root(root, rel)
                if target.exists() and target.is_file():
                    target.unlink()
                    deleted += 1
            except Exception:
                continue

    return {"ok": True, "restored": restored, "deleted": deleted}


class HunkPayload(BaseModel):
    before: str = ""
    after: str = ""
    old: str
    new: str


class RevertHunksRequest(BaseModel):
    snapshot_id: int
    project_dir: str
    path: str
    hunks: List[HunkPayload]


@router.post("/runbook/revert-hunks")
async def revert_hunks(body: RevertHunksRequest):
    """
    Revert selected hunks of a file back to the snapshot content.

    This operates purely on text lines. Each hunk includes:
    - before/after context (unchanged lines)
    - old/new core (changed region)
    """
    root = Path(body.project_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="project_dir must be an existing directory")
    if not _allowed_workdir(root):
        raise HTTPException(status_code=403, detail="project_dir is not allowed")

    snapshot = _load_snapshot(body.snapshot_id)
    snap_files = snapshot.get("files") or {}
    if body.path not in snap_files:
        raise HTTPException(status_code=404, detail="file not found in snapshot")

    target = _resolve_under_root(root, body.path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found on disk")

    try:
        current_text = target.read_text(encoding="utf-8")
    except Exception:
        current_text = target.read_text(errors="ignore")

    current_lines = current_text.split("\n")

    def split_lines(s: str) -> List[str]:
        return [] if s == "" else s.split("\n")

    applied = 0
    failed: List[Dict[str, Any]] = []

    for i, hunk in enumerate(body.hunks):
        before = split_lines(hunk.before)
        after = split_lines(hunk.after)
        old = split_lines(hunk.old)
        new = split_lines(hunk.new)

        pattern = before + new + after
        replacement = before + old + after

        def find_once(pat: List[str]) -> Optional[int]:
            if not pat:
                return None
            hits: List[int] = []
            for start in range(0, len(current_lines) - len(pat) + 1):
                if current_lines[start : start + len(pat)] == pat:
                    hits.append(start)
                    if len(hits) > 1:
                        break
            if len(hits) == 1:
                return hits[0]
            return None

        start_idx = find_once(pattern)
        used_pattern = "context"
        if start_idx is None:
            # Fall back to matching core only if unique.
            start_idx = find_once(new)
            used_pattern = "core"

        if start_idx is None:
            failed.append({"index": i, "reason": "pattern_not_found_or_not_unique", "used": used_pattern})
            continue

        if used_pattern == "core":
            current_lines[start_idx : start_idx + len(new)] = old
        else:
            current_lines[start_idx : start_idx + len(pattern)] = replacement
        applied += 1

    new_text = "\n".join(current_lines)
    target.write_text(new_text, encoding="utf-8")

    return {"ok": True, "applied": applied, "failed": failed}
