from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from paperbot.api.auth.password import verify_password
from paperbot.infrastructure.stores.models import PasswordResetTokenModel, UserModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url
from paperbot.domain.user import User


class SqlAlchemyUserStore:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)

    def _to_domain(self, row: UserModel) -> User:
        return User(
            id=row.id,
            email=row.email,
            github_id=row.github_id,
            github_username=row.github_username,
            display_name=row.display_name,
            avatar_url=row.avatar_url,
            is_active=row.is_active,
            created_at=row.created_at,
        )

    def get_by_id(self, user_id: int) -> Optional[User]:
        with self._provider.session() as session:
            row = session.get(UserModel, user_id)
            return self._to_domain(row) if row else None

    def get_by_email(self, email: str) -> Optional[User]:
        with self._provider.session() as session:
            row = session.query(UserModel).filter_by(email=email).first()
            return self._to_domain(row) if row else None

    def get_by_github_id(self, github_id: str) -> Optional[User]:
        with self._provider.session() as session:
            row = session.query(UserModel).filter_by(github_id=github_id).first()
            return self._to_domain(row) if row else None

    def authenticate(self, email: str, password: str) -> Optional[User]:
        """Verify credentials and update last_login in one session. Returns User or None."""
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            row = session.query(UserModel).filter_by(email=email).first()
            if not row or not row.is_active or not verify_password(password, row.hashed_password or ""):
                return None
            row.last_login_at = now
            session.commit()
            return self._to_domain(row)

    def create_email_user(self, *, email: str, hashed_password: str, display_name: Optional[str] = None) -> User:
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            row = UserModel(
                email=email,
                hashed_password=hashed_password,
                display_name=display_name,
                is_active=True,
                created_at=now,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._to_domain(row)

    def create_github_user(self, *, github_id: str, username: str, display_name: str, avatar_url: str) -> User:
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            row = UserModel(
                github_id=github_id,
                github_username=username,
                display_name=display_name,
                avatar_url=avatar_url,
                is_active=True,
                created_at=now,
                last_login_at=now,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._to_domain(row)

    def update_last_login(self, user_id: int) -> None:
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            session.query(UserModel).filter_by(id=user_id).update({"last_login_at": now})
            session.commit()

    def update_password(self, user_id: int, hashed_password: str) -> None:
        with self._provider.session() as session:
            session.query(UserModel).filter_by(id=user_id).update({"hashed_password": hashed_password})
            session.commit()

    def update_profile(self, user_id: int, display_name: Optional[str]) -> None:
        with self._provider.session() as session:
            session.query(UserModel).filter_by(id=user_id).update({"display_name": display_name})
            session.commit()

    def deactivate(self, user_id: int) -> None:
        """Soft-delete: mark is_active=False so the user cannot log in."""
        with self._provider.session() as session:
            session.query(UserModel).filter_by(id=user_id).update({"is_active": False})
            session.commit()

    def reactivate(self, user_id: int) -> None:
        """Re-enable a previously deactivated account."""
        with self._provider.session() as session:
            session.query(UserModel).filter_by(id=user_id).update({"is_active": True})
            session.commit()

    # ── Password reset tokens ────────────────────────────────────────────────

    def create_reset_token(self, user_id: int) -> str:
        """Generate a secure token, store it, and return the raw token string."""
        now = datetime.now(timezone.utc)
        token = secrets.token_urlsafe(32)
        with self._provider.session() as session:
            row = PasswordResetTokenModel(
                user_id=user_id,
                token=token,
                expires_at=now + timedelta(hours=1),
                used=False,
                created_at=now,
            )
            session.add(row)
            session.commit()
        return token

    def get_valid_reset_token(self, token: str) -> Optional[PasswordResetTokenModel]:
        """Return the token row if it exists, is unused, and has not expired."""
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            row = (
                session.query(PasswordResetTokenModel)
                .filter_by(token=token, used=False)
                .first()
            )
            if not row:
                return None
            # Ensure expires_at is timezone-aware for comparison
            expires = row.expires_at
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires < now:
                return None
            # Detach from session so caller can read attributes
            session.expunge(row)
            return row

    def consume_reset_token(self, token: str, new_hashed_password: str) -> bool:
        """Mark the token as used and update the user's password atomically.

        Returns True on success, False if the token is invalid/expired.
        """
        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            row = (
                session.query(PasswordResetTokenModel)
                .filter_by(token=token, used=False)
                .first()
            )
            if not row:
                return False
            expires = row.expires_at
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires < now:
                return False
            row.used = True
            session.query(UserModel).filter_by(id=row.user_id).update(
                {"hashed_password": new_hashed_password}
            )
            session.commit()
            return True
