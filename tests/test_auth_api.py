from __future__ import annotations

import os

from fastapi.testclient import TestClient

from paperbot.api.main import app


client = TestClient(app)


def _set_jwt_secret() -> None:
    if not os.getenv("PAPERBOT_JWT_SECRET"):
        os.environ["PAPERBOT_JWT_SECRET"] = "test-secret-key"


def test_register_login_me_roundtrip(tmp_path, monkeypatch):
    """Happy path: register -> login -> /me returns user profile."""

    _set_jwt_secret()

    email = "test_user@example.com"
    password = "s3cretP@ss"

    # Register
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "display_name": "Tester"},
    )
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["access_token"]
    assert data["user_id"] > 0

    # Login
    r2 = client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    assert r2.status_code == 200, r2.text
    login_data = r2.json()
    token = login_data["access_token"]

    # /me
    r3 = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r3.status_code == 200, r3.text
    me = r3.json()
    assert me["email"] == email


def test_login_wrong_password_rejected():
    _set_jwt_secret()
    email = "wrong_pw@example.com"
    password = "correctpass123"

    # First register
    client.post(
        "/api/auth/register",
        json={"email": email, "password": password},
    )

    # Then attempt login with wrong password
    r = client.post(
        "/api/auth/login",
        json={"email": email, "password": "badpass"},
    )
    assert r.status_code == 401


def test_register_duplicate_email_rejected():
    _set_jwt_secret()
    email = "duplicate@example.com"
    password = "password123"

    r1 = client.post(
        "/api/auth/register",
        json={"email": email, "password": password},
    )
    assert r1.status_code == 201, r1.text

    r2 = client.post(
        "/api/auth/register",
        json={"email": email, "password": password},
    )
    assert r2.status_code == 400

