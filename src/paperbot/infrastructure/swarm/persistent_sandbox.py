from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from ...repro.base_executor import BaseExecutor


@dataclass
class SandboxLease:
    user_key: str
    session_id: str
    executor_type: str
    sandbox_id: Optional[str]
    updated_at: str


class PersistentSandboxManager:
    """Keeps one reusable sandbox executor per user key."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._executors: Dict[str, BaseExecutor] = {}
        self._leases: Dict[str, SandboxLease] = {}

    def get_or_create(
        self,
        *,
        user_key: str,
        session_id: str,
        requested_sandbox_id: Optional[str] = None,
    ) -> Tuple[Optional[BaseExecutor], Optional[str]]:
        key = (user_key or "default").strip() or "default"
        with self._lock:
            existing = self._executors.get(key)
            if existing is not None and existing.available():
                sandbox_id = self._extract_sandbox_id(existing)
                self._leases[key] = self._new_lease(
                    user_key=key,
                    session_id=session_id,
                    executor=existing,
                    sandbox_id=sandbox_id,
                )
                return existing, sandbox_id

        created = self._create_executor(requested_sandbox_id=requested_sandbox_id)
        if created is None:
            return None, None

        sandbox_id = self._extract_sandbox_id(created)
        with self._lock:
            # Double-check: another thread may have created one while we were unlocked.
            existing = self._executors.get(key)
            if existing is not None and existing.available():
                # Discard the one we just created and use the winner.
                cleanup = getattr(created, "cleanup", None)
                if callable(cleanup):
                    cleanup()
                sandbox_id = self._extract_sandbox_id(existing)
                self._leases[key] = self._new_lease(
                    user_key=key,
                    session_id=session_id,
                    executor=existing,
                    sandbox_id=sandbox_id,
                )
                return existing, sandbox_id

            self._executors[key] = created
            self._leases[key] = self._new_lease(
                user_key=key,
                session_id=session_id,
                executor=created,
                sandbox_id=sandbox_id,
            )
        return created, sandbox_id

    def terminate(self, *, user_key: str) -> None:
        key = (user_key or "default").strip() or "default"
        with self._lock:
            executor = self._executors.pop(key, None)
            self._leases.pop(key, None)

        if executor is None:
            return
        cleanup = getattr(executor, "cleanup", None)
        if callable(cleanup):
            cleanup()

    def lease_for_user(self, user_key: str) -> Optional[SandboxLease]:
        key = (user_key or "default").strip() or "default"
        with self._lock:
            lease = self._leases.get(key)
            if lease is None:
                return None
            return SandboxLease(**lease.__dict__)

    def _create_executor(self, *, requested_sandbox_id: Optional[str]) -> Optional[BaseExecutor]:
        from ...repro.e2b_executor import E2BExecutor

        executor = E2BExecutor(keep_alive=True)
        if not executor.available():
            return None

        if requested_sandbox_id:
            if not executor.attach_sandbox(requested_sandbox_id):
                # If reconnect fails, create a new sandbox.
                if not executor.ensure_sandbox():
                    return None
        else:
            if not executor.ensure_sandbox():
                return None
        return executor

    @staticmethod
    def _extract_sandbox_id(executor: BaseExecutor) -> Optional[str]:
        value = getattr(executor, "sandbox_id", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @staticmethod
    def _new_lease(
        *, user_key: str, session_id: str, executor: BaseExecutor, sandbox_id: Optional[str]
    ) -> SandboxLease:
        return SandboxLease(
            user_key=user_key,
            session_id=session_id,
            executor_type=executor.executor_type,
            sandbox_id=sandbox_id,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
