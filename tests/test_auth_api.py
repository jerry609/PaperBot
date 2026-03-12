from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Give each test its own SQLite database so tests don't share state."""
    db_path = tmp_path / "test_auth.db"
    monkeypatch.setenv("PAPERBOT_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("PAPERBOT_JWT_SECRET", "test-secret-key")

    # Re-import app inside each test so stores pick up the new env vars
    import importlib
    import paperbot.infrastructure.stores.sqlalchemy_db as db_mod
    import paperbot.infrastructure.stores.user_store as us_mod
    import paperbot.api.routes.auth as auth_mod

    importlib.reload(db_mod)
    importlib.reload(us_mod)
    importlib.reload(auth_mod)

    # Run migrations on the fresh DB
    from alembic.config import Config
    from alembic import command

    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(alembic_cfg, "head")

    from paperbot.api.main import app
    yield app


def test_register_login_me_roundtrip(isolated_db):
    """Happy path: register -> login -> /me returns user profile."""
    client = TestClient(isolated_db)
    email = "test_user@example.com"
    password = "s3cretP@ss"

    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "display_name": "Tester"},
    )
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["access_token"]
    assert data["user_id"] > 0

    r2 = client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    assert r2.status_code == 200, r2.text
    token = r2.json()["access_token"]

    r3 = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r3.status_code == 200, r3.text
    assert r3.json()["email"] == email


def test_login_wrong_password_rejected(isolated_db):
    client = TestClient(isolated_db)
    email = "wrong_pw@example.com"
    password = "correctpass123"

    client.post("/api/auth/register", json={"email": email, "password": password})

    r = client.post("/api/auth/login", json={"email": email, "password": "badpass"})
    assert r.status_code == 401


def test_register_duplicate_email_rejected(isolated_db):
    client = TestClient(isolated_db)
    email = "duplicate@example.com"
    password = "password123"

    r1 = client.post("/api/auth/register", json={"email": email, "password": password})
    assert r1.status_code == 201, r1.text

    r2 = client.post("/api/auth/register", json={"email": email, "password": password})
    assert r2.status_code == 400
