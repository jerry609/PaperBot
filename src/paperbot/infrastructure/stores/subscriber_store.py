from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import select

from paperbot.application.ports.subscriber_port import SubscriberPort
from paperbot.infrastructure.stores.models import Base, NewsletterSubscriberModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SubscriberStore(SubscriberPort):
    """CRUD operations for newsletter subscribers."""

    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            self._provider.ensure_tables(Base.metadata)

    def add_subscriber(self, email: str) -> Dict[str, Any]:
        email = email.strip().lower()
        with self._provider.session() as session:
            existing = session.execute(
                select(NewsletterSubscriberModel).where(
                    NewsletterSubscriberModel.email == email
                )
            ).scalar_one_or_none()

            if existing:
                if existing.status == "unsubscribed":
                    existing.status = "active"
                    existing.unsub_at = None
                    existing.subscribed_at = _utcnow()
                    session.commit()
                return self._row_to_dict(existing)

            row = NewsletterSubscriberModel(
                email=email,
                status="active",
                unsub_token=uuid4().hex,
                subscribed_at=_utcnow(),
                metadata_json="{}",
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._row_to_dict(row)

    def remove_subscriber(self, unsub_token: str) -> bool:
        with self._provider.session() as session:
            row = session.execute(
                select(NewsletterSubscriberModel).where(
                    NewsletterSubscriberModel.unsub_token == unsub_token
                )
            ).scalar_one_or_none()
            if not row:
                return False
            if row.status == "unsubscribed":
                return True
            row.status = "unsubscribed"
            row.unsub_at = _utcnow()
            session.commit()
            return True

    def get_active_subscribers(self) -> List[str]:
        with self._provider.session() as session:
            rows = session.execute(
                select(NewsletterSubscriberModel).where(
                    NewsletterSubscriberModel.status == "active"
                )
            ).scalars().all()
            return [r.email for r in rows]

    def get_active_subscribers_with_tokens(self) -> Dict[str, str]:
        with self._provider.session() as session:
            rows = session.execute(
                select(NewsletterSubscriberModel).where(
                    NewsletterSubscriberModel.status == "active"
                )
            ).scalars().all()
            return {r.email: r.unsub_token for r in rows}

    def get_subscriber_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        email = email.strip().lower()
        with self._provider.session() as session:
            row = session.execute(
                select(NewsletterSubscriberModel).where(
                    NewsletterSubscriberModel.email == email
                )
            ).scalar_one_or_none()
            if not row:
                return None
            return self._row_to_dict(row)

    def get_subscriber_count(self) -> Dict[str, int]:
        with self._provider.session() as session:
            all_rows = session.execute(
                select(NewsletterSubscriberModel)
            ).scalars().all()
            active = sum(1 for r in all_rows if r.status == "active")
            return {"active": active, "total": len(all_rows)}

    @staticmethod
    def _row_to_dict(row: NewsletterSubscriberModel) -> Dict[str, Any]:
        return {
            "id": row.id,
            "email": row.email,
            "status": row.status,
            "unsub_token": row.unsub_token,
            "subscribed_at": row.subscribed_at.isoformat() if row.subscribed_at else None,
            "unsub_at": row.unsub_at.isoformat() if row.unsub_at else None,
        }
