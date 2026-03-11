"""SubscriberPort — newsletter subscriber management interface."""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class SubscriberPort(Protocol):
    """Abstract interface for newsletter subscriber operations."""

    def add_subscriber(self, email: str) -> Dict[str, Any]: ...

    def remove_subscriber(self, unsub_token: str) -> bool: ...

    def get_subscriber_count(self) -> Dict[str, int]: ...
