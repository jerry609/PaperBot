from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import select

from paperbot.infrastructure.stores.keychain import KeychainStore
from paperbot.infrastructure.stores.models import Base, EmbeddingEndpointModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url
from paperbot.utils.secret import decrypt as _decrypt_secret
from paperbot.utils.secret import encrypt as _encrypt_secret

_KEYCHAIN_MARKER = "__keychain__"
_DEFAULT_SCOPE = "default"
_DEFAULT_PROVIDER = "openai"
_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_API_KEY_ENV = "PAPERBOT_EMBEDDING_API_KEY"
_ALLOWED_PROVIDERS = {"openai"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _keychain_name(scope: str) -> str:
    normalized = str(scope or _DEFAULT_SCOPE).strip() or _DEFAULT_SCOPE
    return f"embedding-endpoint:{normalized}"


def _mask_secret(value: str) -> str:
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= 8:
        return "***"
    return f"***{text[-8:]}"


class EmbeddingEndpointStore:
    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            self._provider.ensure_tables(Base.metadata)

    def get_config(self, *, include_secrets: bool = False) -> Optional[Dict[str, Any]]:
        with self._provider.session() as session:
            row = session.execute(
                select(EmbeddingEndpointModel).where(EmbeddingEndpointModel.scope == _DEFAULT_SCOPE)
            ).scalar_one_or_none()
            return self._to_dict(row, include_secrets=include_secrets) if row else None

    def upsert_config(self, *, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = _utcnow()
        with self._provider.session() as session:
            row = session.execute(
                select(EmbeddingEndpointModel).where(EmbeddingEndpointModel.scope == _DEFAULT_SCOPE)
            ).scalar_one_or_none()
            creating = row is None
            if row is None:
                row = EmbeddingEndpointModel(
                    scope=_DEFAULT_SCOPE,
                    provider=_DEFAULT_PROVIDER,
                    api_key_env=_DEFAULT_API_KEY_ENV,
                    model=_DEFAULT_MODEL,
                    enabled=False,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)

            provider = (
                str(payload.get("provider") or row.provider or _DEFAULT_PROVIDER).strip().lower()
            )
            if provider not in _ALLOWED_PROVIDERS:
                raise ValueError(f"unsupported embedding provider: {provider}")

            model = str(payload.get("model") or row.model or _DEFAULT_MODEL).strip()
            if not model:
                raise ValueError("embedding model is required")

            row.provider = provider
            row.base_url = str(payload.get("base_url") or row.base_url or "").strip() or None
            row.api_key_env = (
                str(payload.get("api_key_env") or row.api_key_env or _DEFAULT_API_KEY_ENV).strip()
                or _DEFAULT_API_KEY_ENV
            )
            row.model = model
            row.enabled = bool(payload.get("enabled", row.enabled))
            row.updated_at = now
            if creating:
                row.created_at = now

            if "api_key" in payload:
                api_key_text = str(payload.get("api_key") or "").strip()
                if not api_key_text:
                    row.api_key_value = None
                    KeychainStore.delete_key(_keychain_name(row.scope))
                elif not api_key_text.startswith("***"):
                    if KeychainStore.store_key(_keychain_name(row.scope), api_key_text):
                        row.api_key_value = _KEYCHAIN_MARKER
                    else:
                        row.api_key_value = _encrypt_secret(api_key_text)

            session.commit()
            session.refresh(row)
            return self._to_dict(row)

    @staticmethod
    def _to_dict(
        row: EmbeddingEndpointModel,
        *,
        include_secrets: bool = False,
    ) -> Dict[str, Any]:
        stored = str(row.api_key_value or "").strip()
        if stored == _KEYCHAIN_MARKER:
            key_raw = KeychainStore.get_key(_keychain_name(row.scope)) or ""
            key_source = "keychain"
        else:
            key_raw = _decrypt_secret(stored)
            key_source = "db" if key_raw else ""

        api_key_env = str(row.api_key_env or _DEFAULT_API_KEY_ENV)
        key_present = bool(key_raw)
        key_display = key_raw if include_secrets else _mask_secret(key_raw)

        return {
            "id": int(row.id),
            "scope": row.scope,
            "provider": row.provider,
            "base_url": row.base_url,
            "api_key_env": api_key_env,
            "api_key": key_display,
            "api_key_present": key_present,
            "key_source": key_source,
            "model": row.model,
            "enabled": bool(row.enabled),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def close(self) -> None:
        try:
            self._provider.engine.dispose()
        except Exception:
            pass
