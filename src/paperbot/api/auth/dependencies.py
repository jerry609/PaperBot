from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError

from paperbot.api.auth.jwt import decode_token
from paperbot.infrastructure.stores.user_store import SqlAlchemyUserStore

logger = logging.getLogger(__name__)

bearer = HTTPBearer(auto_error=False)

AUTH_OPTIONAL = os.getenv("AUTH_OPTIONAL", "false").lower() in {"1", "true", "yes"}

_user_store = SqlAlchemyUserStore()


def _resolve_user(credentials: Optional[HTTPAuthorizationCredentials]):
    if not credentials or not credentials.credentials:
        logger.warning("[auth] Missing token — no credentials provided")
        if AUTH_OPTIONAL:
            return None
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    token = credentials.credentials
    logger.debug("[auth] Received token (first 20 chars): %s...", token[:20])
    try:
        user_id = decode_token(token)
        logger.debug("[auth] Token valid, user_id=%s", user_id)
    except JWTError as e:
        logger.warning("[auth] Invalid token: %s | token prefix: %s...", e, token[:20])
        if AUTH_OPTIONAL:
            return None
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = _user_store.get_by_id(user_id)
    if not user or not user.is_active:
        if AUTH_OPTIONAL:
            return None
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
):
    """Strict user dependency: always requires a valid user.

    Use this for endpoints that must not fall back to the legacy "default" namespace.
    """

    user = _resolve_user(credentials)
    if user is None:
        # AUTH_OPTIONAL only affects get_user_id; this dependency always enforces auth.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")
    return user


def get_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> str:
    """Return the authenticated user id as a string.

    When AUTH_OPTIONAL=true, missing/invalid tokens fall back to "default" so
    legacy callers keep functioning while we migrate to multi-user auth.
    """

    user = _resolve_user(credentials)
    if user is None:
        return "default"
    return str(user.id)
