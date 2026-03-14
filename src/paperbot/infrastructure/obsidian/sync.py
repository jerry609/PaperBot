from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple
from uuid import uuid4

from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.application.ports.memory_port import MemoryPort
from paperbot.memory.schema import MemoryCandidate
from paperbot.utils.user_identity import optional_user_identity

from .conflict import ObsidianManagedConflict, detect_managed_conflict
from .parser import ParsedObsidianNote, parse_note_text, user_sections_hash

PAPER_MANAGED_HEADINGS: tuple[str, ...] = (
    "Summary",
    "Metadata",
    "Tracks",
    "Related Papers",
    "References",
    "Cited By",
    "Links",
)
TRACK_MANAGED_HEADINGS: tuple[str, ...] = (
    "Focus",
    "Research Tasks",
    "Milestones",
    "Tracked Scholars",
    "Saved Papers",
)


@dataclass(frozen=True)
class ObsidianSyncStatus:
    last_synced_at: Optional[str]
    pending_count: int
    tracked_note_count: int
    conflict_count: int
    state_path: str
    pending_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObsidianSyncScanResult:
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObsidianBidirectionalSync:
    def __init__(
        self,
        *,
        vault_path: Path,
        root_dir: str,
        memory_store: MemoryPort,
        event_log: EventLogPort,
        sync_state_filename: str = ".paperbot-sync-state.json",
        pending_dirname: str = ".paperbot-pending",
    ) -> None:
        self._vault_path = Path(vault_path).expanduser()
        self._root_dir = str(root_dir or "PaperBot").strip() or "PaperBot"
        self._memory_store = memory_store
        self._event_log = event_log
        self._sync_state_filename = str(sync_state_filename or ".paperbot-sync-state.json")
        self._pending_dirname = str(pending_dirname or ".paperbot-pending")

    @property
    def root_path(self) -> Path:
        return self._vault_path / self._root_dir

    @property
    def state_path(self) -> Path:
        return self.root_path / self._sync_state_filename

    @property
    def pending_dir(self) -> Path:
        return self.root_path / self._pending_dirname

    def get_status(self) -> ObsidianSyncStatus:
        self._ensure_root_path()
        state = self._load_state()
        return ObsidianSyncStatus(
            last_synced_at=_coerce_optional_string(state.get("last_synced_at")),
            pending_count=self._pending_count(),
            tracked_note_count=len(self._notes_state(state)),
            conflict_count=len(self._conflicts_state(state)),
            state_path=str(self.state_path),
            pending_dir=str(self.pending_dir),
        )

    def scan(self) -> ObsidianSyncScanResult:
        return self._scan(paths=None)

    def sync_paths(self, paths: Sequence[Path]) -> ObsidianSyncScanResult:
        return self._scan(paths=paths)

    def _scan(self, paths: Optional[Sequence[Path]]) -> ObsidianSyncScanResult:
        self._ensure_root_path()
        state = self._load_state()
        notes_state = self._notes_state(state)
        conflicts_state = self._conflicts_state(state)
        run_id = f"obsidian-sync-{uuid4().hex[:12]}"
        last_synced_at = datetime.now(timezone.utc).isoformat()

        scanned_notes = 0
        changed_notes = 0
        memories_created = 0
        memories_skipped = 0
        tag_updates = 0
        wikilink_updates = 0
        note_updates = 0
        conflicts_detected = 0

        candidate_paths = self._resolve_candidate_paths(paths)
        seen_rel_paths: set[str] = set()
        for note_path in candidate_paths:
            relative_path = self._relative_note_path(note_path)
            seen_rel_paths.add(relative_path)
            previous = notes_state.get(relative_path)

            if not note_path.exists():
                if previous is not None:
                    changed_notes += 1
                    notes_state.pop(relative_path, None)
                    conflicts_state.pop(relative_path, None)
                    self._append_event(
                        run_id=run_id,
                        event_type="obsidian.note.deleted",
                        payload={"path": relative_path},
                    )
                continue

            if not note_path.is_file() or note_path.suffix.lower() != ".md":
                continue

            parsed_note = self._parse_note(note_path=note_path, run_id=run_id)
            if parsed_note is None:
                continue

            paperbot_type = str(parsed_note.frontmatter.get("paperbot_type") or "").strip().lower()
            if paperbot_type not in {"paper", "track"}:
                continue

            scanned_notes += 1
            current_snapshot = self._snapshot_for_note(note_path=note_path, note=parsed_note)
            if previous == current_snapshot:
                continue

            changed_notes += 1
            notes_state[relative_path] = current_snapshot

            conflict = detect_managed_conflict(
                note_path=note_path,
                frontmatter=parsed_note.frontmatter,
                managed_body=parsed_note.managed_body,
            )
            if conflict is not None:
                conflicts_state[relative_path] = conflict.to_dict()
                conflicts_detected += 1
                self._append_conflict_event(run_id=run_id, conflict=conflict, path=relative_path)
            else:
                conflicts_state.pop(relative_path, None)

            if previous is None:
                self._append_event(
                    run_id=run_id,
                    event_type="obsidian.note.baseline_recorded",
                    payload={
                        "path": relative_path,
                        "paperbot_type": paperbot_type,
                    },
                )
                if parsed_note.user_sections:
                    created_count, skipped_count = self._sync_user_notes(
                        note=parsed_note,
                        path=relative_path,
                    )
                    memories_created += created_count
                    memories_skipped += skipped_count
                    if created_count or skipped_count:
                        note_updates += 1
                continue

            added_tags = _difference(parsed_note.user_tags, previous.get("tags"))
            added_wikilinks = _difference(parsed_note.user_wikilinks, previous.get("wikilinks"))
            notes_changed = (
                previous.get("user_sections_hash") != current_snapshot["user_sections_hash"]
            )

            created_count, skipped_count = self._sync_note_delta(
                note=parsed_note,
                path=relative_path,
                added_tags=added_tags,
                added_wikilinks=added_wikilinks,
                user_notes_changed=notes_changed,
            )
            memories_created += created_count
            memories_skipped += skipped_count
            tag_updates += len(added_tags)
            wikilink_updates += len(added_wikilinks)
            note_updates += 1 if notes_changed else 0

        if paths is None:
            missing_paths = [path for path in list(notes_state) if path not in seen_rel_paths]
            for missing_path in missing_paths:
                notes_state.pop(missing_path, None)
                conflicts_state.pop(missing_path, None)

        state["last_synced_at"] = last_synced_at
        self._save_state(state)
        return ObsidianSyncScanResult(
            last_synced_at=last_synced_at,
            scanned_notes=scanned_notes,
            changed_notes=changed_notes,
            memories_created=memories_created,
            memories_skipped=memories_skipped,
            tag_updates=tag_updates,
            wikilink_updates=wikilink_updates,
            note_updates=note_updates,
            conflicts_detected=conflicts_detected,
            pending_count=self._pending_count(),
        )

    def _sync_note_delta(
        self,
        *,
        note: ParsedObsidianNote,
        path: str,
        added_tags: Sequence[str],
        added_wikilinks: Sequence[str],
        user_notes_changed: bool,
    ) -> Tuple[int, int]:
        created_total = 0
        skipped_total = 0
        user_id, scope_type, scope_id = self._scope_for_note(note)
        title = note.title or Path(path).stem

        if added_tags:
            memories = [
                MemoryCandidate(
                    kind="keyword_set",
                    content=f'Obsidian tag for "{title}": {tag}',
                    confidence=0.85,
                    tags=["obsidian", "tag", tag],
                    evidence={"path": path, "tag": tag},
                    scope_type=scope_type,
                    scope_id=scope_id,
                    status="approved",
                )
                for tag in added_tags
            ]
            created, skipped = self._write_memories(user_id=user_id, memories=memories)
            created_total += created
            skipped_total += skipped
            for tag in added_tags:
                self._append_event(
                    run_id=f"obsidian-sync-tags-{uuid4().hex[:10]}",
                    event_type="obsidian.note.tag_added",
                    payload={"path": path, "tag": tag, "title": title},
                )

        if added_wikilinks:
            memories = [
                MemoryCandidate(
                    kind="fact",
                    content=f'Obsidian wiki-link from "{title}" to "{target}"',
                    confidence=0.8,
                    tags=["obsidian", "wikilink"],
                    evidence={"path": path, "target": target},
                    scope_type=scope_type,
                    scope_id=scope_id,
                    status="approved",
                )
                for target in added_wikilinks
            ]
            created, skipped = self._write_memories(user_id=user_id, memories=memories)
            created_total += created
            skipped_total += skipped
            for target in added_wikilinks:
                self._append_event(
                    run_id=f"obsidian-sync-links-{uuid4().hex[:10]}",
                    event_type="obsidian.note.wikilink_added",
                    payload={"path": path, "target": target, "title": title},
                )

        if user_notes_changed:
            created, skipped = self._sync_user_notes(note=note, path=path)
            created_total += created
            skipped_total += skipped

        return created_total, skipped_total

    def _sync_user_notes(self, *, note: ParsedObsidianNote, path: str) -> Tuple[int, int]:
        if not note.user_sections:
            return 0, 0

        user_id, scope_type, scope_id = self._scope_for_note(note)
        title = note.title or Path(path).stem
        memories: List[MemoryCandidate] = []
        for section in note.user_sections:
            content = _section_content(section.markdown)
            if not content:
                continue
            memories.append(
                MemoryCandidate(
                    kind="note",
                    content=f'Obsidian personal note for "{title}" ({section.heading}): {content}',
                    confidence=0.7,
                    tags=["obsidian", "personal-note"],
                    evidence={"path": path, "heading": section.heading},
                    scope_type=scope_type,
                    scope_id=scope_id,
                    status="approved",
                )
            )

        if not memories:
            return 0, 0

        created, skipped = self._write_memories(user_id=user_id, memories=memories)
        self._append_event(
            run_id=f"obsidian-sync-notes-{uuid4().hex[:10]}",
            event_type="obsidian.note.user_sections_synced",
            payload={"path": path, "sections": [section.heading for section in note.user_sections]},
        )
        return created, skipped

    def _write_memories(
        self, *, user_id: Optional[str], memories: List[MemoryCandidate]
    ) -> Tuple[int, int]:
        resolved_user_id = optional_user_identity(user_id)
        if resolved_user_id is None:
            return 0, 0
        created, skipped, _ = self._memory_store.add_memories(
            user_id=resolved_user_id,
            memories=memories,
            actor_id="obsidian-sync",
        )
        return created, skipped

    def _scope_for_note(self, note: ParsedObsidianNote) -> tuple[Optional[str], str, Optional[str]]:
        frontmatter = note.frontmatter
        user_id = optional_user_identity(frontmatter.get("user_id"))
        paperbot_type = str(frontmatter.get("paperbot_type") or "").strip().lower()
        if paperbot_type == "track":
            scope_id = _coerce_optional_string(frontmatter.get("track_id")) or (
                _coerce_optional_string(frontmatter.get("name")) or note.title or "track"
            )
            return user_id, "track", scope_id

        if paperbot_type == "paper":
            scope_id = (
                _coerce_optional_string(frontmatter.get("paperbot_id"))
                or _coerce_optional_string(frontmatter.get("semantic_scholar_id"))
                or _coerce_optional_string(frontmatter.get("doi"))
                or _coerce_optional_string(frontmatter.get("arxiv_id"))
                or _coerce_optional_string(frontmatter.get("openalex_id"))
                or note.title
                or "paper"
            )
            return user_id, "paper", scope_id

        return user_id, "global", None

    def _parse_note(self, *, note_path: Path, run_id: str) -> Optional[ParsedObsidianNote]:
        try:
            text = note_path.read_text(encoding="utf-8")
            managed_headings = self._managed_headings_for_type(note_path=note_path, text=text)
            return parse_note_text(text, managed_headings=managed_headings)
        except Exception as exc:
            self._append_event(
                run_id=run_id,
                event_type="obsidian.note.parse_failed",
                payload={"path": self._relative_note_path(note_path), "error": str(exc)},
            )
            return None

    def _managed_headings_for_type(self, *, note_path: Path, text: str) -> Sequence[str]:
        try:
            note = parse_note_text(text)
        except Exception:
            return ()
        paperbot_type = str(note.frontmatter.get("paperbot_type") or "").strip().lower()
        if paperbot_type == "paper":
            return PAPER_MANAGED_HEADINGS
        if paperbot_type == "track":
            return TRACK_MANAGED_HEADINGS
        if note_path.name == "MOC.md":
            return ("Tracks", "Papers")
        return ()

    def _resolve_candidate_paths(self, paths: Optional[Sequence[Path]]) -> List[Path]:
        if paths is None:
            return sorted(self._iter_note_paths())

        resolved: List[Path] = []
        for raw_path in paths:
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = (self.root_path / candidate).resolve()
            resolved.append(candidate)
        return sorted({path for path in resolved})

    def _iter_note_paths(self) -> Iterable[Path]:
        pending_prefix = self.pending_dir.resolve()
        for note_path in self.root_path.rglob("*.md"):
            resolved = note_path.resolve()
            if resolved == self.state_path.resolve():
                continue
            if pending_prefix in resolved.parents:
                continue
            yield resolved

    def _snapshot_for_note(self, *, note_path: Path, note: ParsedObsidianNote) -> Dict[str, Any]:
        return {
            "mtime_ns": note_path.stat().st_mtime_ns,
            "paperbot_type": str(note.frontmatter.get("paperbot_type") or "").strip().lower(),
            "managed_hash": _coerce_optional_string(note.frontmatter.get("paperbot_managed_hash")),
            "tags": list(note.user_tags),
            "wikilinks": list(note.user_wikilinks),
            "user_sections_hash": user_sections_hash(note.user_sections),
        }

    def _append_conflict_event(
        self,
        *,
        run_id: str,
        conflict: ObsidianManagedConflict,
        path: str,
    ) -> None:
        self._append_event(
            run_id=run_id,
            event_type="obsidian.note.conflict_detected",
            payload={
                "path": path,
                "reason": conflict.reason,
                "expected_hash": conflict.expected_hash,
                "actual_hash": conflict.actual_hash,
                "detected_at": conflict.detected_at,
            },
        )

    def _append_event(self, *, run_id: str, event_type: str, payload: Mapping[str, Any]) -> None:
        self._event_log.append(
            {
                "run_id": run_id,
                "trace_id": run_id,
                "workflow": "obsidian_sync",
                "stage": "scan",
                "agent_name": "obsidian-sync",
                "role": "system",
                "type": event_type,
                "payload": dict(payload),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _ensure_root_path(self) -> None:
        if not self._vault_path.exists() or not self._vault_path.is_dir():
            raise ValueError("vault_path must be an existing directory")
        if not self.root_path.exists() or not self.root_path.is_dir():
            raise ValueError("obsidian root_dir must already exist inside the vault")

    def _pending_count(self) -> int:
        if not self.pending_dir.exists():
            return 0
        return sum(1 for _ in self.pending_dir.rglob("*.md"))

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"last_synced_at": None, "notes": {}, "conflicts": {}}
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Obsidian sync state file must contain a JSON object")
        payload.setdefault("last_synced_at", None)
        payload.setdefault("notes", {})
        payload.setdefault("conflicts", {})
        return payload

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _notes_state(self, state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        notes = state.setdefault("notes", {})
        if not isinstance(notes, dict):
            raise ValueError("Obsidian sync notes state must be a JSON object")
        return notes

    def _conflicts_state(self, state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        conflicts = state.setdefault("conflicts", {})
        if not isinstance(conflicts, dict):
            raise ValueError("Obsidian sync conflicts state must be a JSON object")
        return conflicts

    def _relative_note_path(self, note_path: Path) -> str:
        return str(note_path.resolve().relative_to(self.root_path.resolve()))


def _coerce_optional_string(value: Any) -> Optional[str]:
    normalized = str(value or "").strip()
    return normalized or None


def _difference(current: Sequence[str], previous: Any) -> List[str]:
    previous_set = {
        str(item).strip().casefold() for item in list(previous or []) if str(item).strip()
    }
    return [item for item in current if item.casefold() not in previous_set]


def _section_content(markdown: str) -> str:
    lines = str(markdown or "").splitlines()
    if lines and lines[0].startswith("## "):
        lines = lines[1:]
    content = "\n".join(lines).strip()
    return content[:1000]
