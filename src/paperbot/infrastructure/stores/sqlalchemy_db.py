from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session


DEFAULT_DB_URL = "sqlite:///data/paperbot.db"


def get_db_url() -> str:
    return os.getenv("PAPERBOT_DB_URL") or DEFAULT_DB_URL


def _ensure_sqlite_parent_dir(db_url: str) -> None:
    # sqlite:///relative/path.db or sqlite:////abs/path.db
    if not db_url.startswith("sqlite:"):
        return
    if db_url.startswith("sqlite:///:"):
        path = db_url.replace("sqlite:///", "", 1)
    elif db_url.startswith("sqlite:////"):
        path = db_url.replace("sqlite:////", "/", 1)
    else:
        # sqlite:// (rare) or sqlite:pure-memory
        return
    if path in (":memory:", ""):
        return
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def create_db_engine(db_url: Optional[str] = None) -> Engine:
    url = db_url or get_db_url()
    _ensure_sqlite_parent_dir(url)
    connect_args = {}
    if url.startswith("sqlite:"):
        connect_args = {"check_same_thread": False}
    return create_engine(url, future=True, pool_pre_ping=True, connect_args=connect_args)


def create_session_factory(engine: Engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class SessionProvider:
    """Light wrapper to create/close SQLAlchemy sessions."""

    def __init__(self, db_url: Optional[str] = None):
        self.engine = create_db_engine(db_url)
        self._factory = create_session_factory(self.engine)

    def session(self) -> Session:
        return self._factory()


