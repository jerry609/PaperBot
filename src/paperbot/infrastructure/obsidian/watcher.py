from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in environments without watchdog installed
    FileSystemEvent = object  # type: ignore[assignment]
    FileSystemEventHandler = object  # type: ignore[assignment]
    Observer = None  # type: ignore[assignment]
    WATCHDOG_AVAILABLE = False


class DebouncedPathQueue:
    def __init__(
        self,
        *,
        debounce_seconds: float,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._debounce_seconds = max(0.0, float(debounce_seconds))
        self._time_fn = time_fn or time.monotonic
        self._pending: Dict[str, float] = {}
        self._lock = threading.Lock()

    def push(self, path: Path) -> None:
        normalized = str(Path(path).expanduser())
        if not normalized:
            return
        with self._lock:
            self._pending[normalized] = self._time_fn() + self._debounce_seconds

    def flush_ready(self) -> List[Path]:
        now = self._time_fn()
        ready: List[Path] = []
        with self._lock:
            for raw_path, ready_at in list(self._pending.items()):
                if ready_at > now:
                    continue
                ready.append(Path(raw_path))
                self._pending.pop(raw_path, None)
        return sorted(ready)

    def flush_all(self) -> List[Path]:
        with self._lock:
            ready = [Path(raw_path) for raw_path in self._pending]
            self._pending.clear()
        return sorted(ready)


class ObsidianVaultWatcher:
    def __init__(
        self,
        *,
        root_path: Path,
        on_paths_changed: Callable[[List[Path]], object],
        debounce_seconds: float = 1.0,
        poll_interval_seconds: float = 0.25,
    ) -> None:
        self._root_path = Path(root_path).expanduser().resolve()
        self._on_paths_changed = on_paths_changed
        self._debounce_queue = DebouncedPathQueue(debounce_seconds=debounce_seconds)
        self._poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self._observer: Optional[Observer] = None
        self._stop_event = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None
        self._is_running = False

    @property
    def is_available(self) -> bool:
        return WATCHDOG_AVAILABLE

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def root_path(self) -> Path:
        return self._root_path

    def start(self) -> bool:
        if self._is_running:
            return True
        if not WATCHDOG_AVAILABLE or Observer is None:
            return False
        if not self._root_path.exists() or not self._root_path.is_dir():
            raise ValueError("Obsidian watcher root_path must be an existing directory")

        self._stop_event.clear()
        self._observer = Observer()
        self._observer.schedule(_WatchdogEventHandler(self), str(self._root_path), recursive=True)
        self._observer.start()

        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        self._is_running = True
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=2.0)
            self._flush_thread = None
        self._is_running = False

    def record_event(self, path: Path) -> None:
        candidate = Path(path).expanduser()
        if candidate.suffix.lower() != ".md":
            return
        self._debounce_queue.push(
            candidate.resolve()
            if candidate.is_absolute()
            else (self._root_path / candidate).resolve()
        )

    def flush_pending(self) -> List[Path]:
        ready_paths = self._debounce_queue.flush_all()
        if ready_paths:
            self._on_paths_changed(ready_paths)
        return ready_paths

    def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            ready_paths = self._debounce_queue.flush_ready()
            if ready_paths:
                self._on_paths_changed(ready_paths)
            self._stop_event.wait(self._poll_interval_seconds)


class _WatchdogEventHandler(FileSystemEventHandler):
    def __init__(self, watcher: ObsidianVaultWatcher) -> None:
        super().__init__()
        self._watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:
        self._record_event(event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._record_event(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._record_event(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        src_path = getattr(event, "src_path", None)
        dest_path = getattr(event, "dest_path", None)
        if src_path:
            self._watcher.record_event(Path(str(src_path)))
        if dest_path:
            self._watcher.record_event(Path(str(dest_path)))

    def _record_event(self, event: FileSystemEvent) -> None:
        if getattr(event, "is_directory", False):
            return
        src_path = getattr(event, "src_path", None)
        if src_path:
            self._watcher.record_event(Path(str(src_path)))
