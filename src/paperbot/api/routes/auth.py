from __future__ import annotations

import os
import httpx
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

try:
    import email_validator  # noqa: F401
    from pydantic import EmailStr
except Exception:  # pragma: no cover - exercised only in minimal test envs
    EmailStr = str  # type: ignore[assignment]

from paperbot.api.auth.password import hash_password
from paperbot.api.auth.jwt import create_access_token
from paperbot.api.auth.dependencies import get_current_user
from paperbot.api.auth.email import send_password_reset_email
from paperbot.infrastructure.stores.user_store import SqlAlchemyUserStore
from paperbot.domain.user import User


router = APIRouter(prefix="/api/auth", tags=["auth"])

_user_store = SqlAlchemyUserStore()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    display_name: Optional[str]


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(req: RegisterRequest):
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password too short (min 8 chars)")

    existing = _user_store.get_by_email(req.email)
    if existing:
        if existing.is_active:
            raise HTTPException(status_code=400, detail="Email already registered")
        # Deactivated account: reactivate and reset password
        _user_store.reactivate(existing.id)
        _user_store.update_password(existing.id, hash_password(req.password))
        if req.display_name:
            _user_store.update_profile(existing.id, req.display_name)
        return TokenResponse(
            access_token=create_access_token(existing.id),
            user_id=existing.id,
            display_name=req.display_name or existing.display_name,
        )

    user = _user_store.create_email_user(
        email=req.email,
        hashed_password=hash_password(req.password),
        display_name=req.display_name,
    )
    return TokenResponse(
        access_token=create_access_token(user.id),
        user_id=user.id,
        display_name=user.display_name,
    )


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    existing = _user_store.get_by_email(req.email)
    if not existing:
        raise HTTPException(status_code=401, detail="Email not registered.")
    if not existing.is_active:
        raise HTTPException(status_code=401, detail="Account has been deleted. Please register again.")
    user = _user_store.authenticate(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect password.")
    return TokenResponse(
        access_token=create_access_token(user.id),
        user_id=user.id,
        display_name=user.display_name,
    )


class MeResponse(BaseModel):
    id: int
    email: Optional[str]
    github_username: Optional[str]
    display_name: Optional[str]
    avatar_url: Optional[str]


@router.get("/me", response_model=MeResponse)
def me(current_user: User = Depends(get_current_user)):
    return MeResponse(
        id=current_user.id,
        email=current_user.email,
        github_username=current_user.github_username,
        display_name=current_user.display_name,
        avatar_url=current_user.avatar_url,
    )


class GithubExchangeRequest(BaseModel):
    github_id: str
    login: Optional[str] = None
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    email: Optional[str] = None
    access_token: str


@router.post("/github/exchange", response_model=TokenResponse)
async def github_exchange(req: GithubExchangeRequest):
    # Verify the GitHub token against GitHub API to prevent forged requests
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {req.access_token}", "Accept": "application/json"},
            timeout=15.0,
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Invalid GitHub token")
    gh = resp.json()
    if str(gh.get("id")) != str(req.github_id):
        raise HTTPException(status_code=400, detail="GitHub id mismatch")

    user = _user_store.get_by_github_id(req.github_id)
    if user:
        if not user.is_active:
            _user_store.reactivate(user.id)
        _user_store.update_last_login(user.id)
    else:
        user = _user_store.create_github_user(
            github_id=str(req.github_id),
            username=req.login or (gh.get("login") or ""),
            display_name=(req.name or gh.get("name") or gh.get("login") or ""),
            avatar_url=(req.avatar_url or gh.get("avatar_url") or ""),
        )

    return TokenResponse(
        access_token=create_access_token(user.id),
        user_id=user.id,
        display_name=user.display_name,
    )


# ── Account management ───────────────────────────────────────────────────────

class UpdateMeRequest(BaseModel):
    display_name: Optional[str] = None


@router.patch("/me", response_model=MeResponse)
def update_me(req: UpdateMeRequest, current_user: User = Depends(get_current_user)):
    update_data = req.model_dump(exclude_unset=True)
    if "display_name" in update_data:
        _user_store.update_profile(current_user.id, update_data["display_name"])
    updated_user = _user_store.get_by_id(current_user.id)
    return MeResponse(
        id=current_user.id,
        email=current_user.email,
        github_username=current_user.github_username,
        display_name=updated_user.display_name if updated_user else current_user.display_name,
        avatar_url=current_user.avatar_url,
    )


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.post("/me/change-password", status_code=200)
def change_password(req: ChangePasswordRequest, current_user: User = Depends(get_current_user)):
    if not current_user.email:
        raise HTTPException(status_code=400, detail="Password change not available for OAuth accounts")
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password too short (min 8 chars)")
    if not _user_store.authenticate(current_user.email, req.current_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    _user_store.update_password(current_user.id, hash_password(req.new_password))
    return {"detail": "Password updated."}


@router.delete("/me", status_code=204)
def delete_me(current_user: User = Depends(get_current_user)):
    _user_store.deactivate(current_user.id)


# ── Forgot / Reset password ──────────────────────────────────────────────────

class ForgotPasswordRequest(BaseModel):
    email: EmailStr


@router.post("/forgot-password", status_code=202)
def forgot_password(req: ForgotPasswordRequest):
    """Send a password-reset link. Always returns 202 to avoid email enumeration."""
    user = _user_store.get_by_email(req.email)
    if user and user.is_active and user.email:
        token = _user_store.create_reset_token(user.id)
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        reset_url = f"{frontend_url}/reset-password?token={token}"
        send_password_reset_email(user.email, reset_url)
    return {"detail": "If that email is registered, a reset link has been sent."}


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


@router.post("/reset-password", status_code=200)
def reset_password(req: ResetPasswordRequest):
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password too short (min 8 chars)")
    ok = _user_store.consume_reset_token(req.token, hash_password(req.new_password))
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    return {"detail": "Password updated successfully."}
