from __future__ import annotations

from typing import Optional

LEGACY_DEFAULT_USER_ID = "default"


def normalize_user_id(user_id: Optional[str]) -> Optional[str]:
    value = str(user_id or "").strip()
    return value or None


def optional_user_identity(user_id: Optional[str]) -> Optional[str]:
    value = normalize_user_id(user_id)
    if value == LEGACY_DEFAULT_USER_ID:
        return None
    return value


def has_user_identity(user_id: Optional[str]) -> bool:
    return optional_user_identity(user_id) is not None


def require_user_identity(user_id: Optional[str]) -> str:
    value = optional_user_identity(user_id)
    if value is None:
        normalized = normalize_user_id(user_id)
        if normalized == LEGACY_DEFAULT_USER_ID:
            raise ValueError("legacy user_id 'default' is no longer supported")
        raise ValueError("user_id is required")
    return value
