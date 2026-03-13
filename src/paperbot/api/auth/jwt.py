import os
from datetime import datetime, timedelta, timezone

try:
    from jose import jwt
except ImportError:  # pragma: no cover - exercised only in minimal test envs
    jwt = None  # type: ignore[assignment]

SECRET_KEY = os.environ.get("PAPERBOT_JWT_SECRET", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days


def create_access_token(user_id: int) -> str:
    if jwt is None:
        raise RuntimeError("python-jose is required for JWT auth. Install with `pip install python-jose`.")
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> int:
    if jwt is None:
        raise RuntimeError("python-jose is required for JWT auth. Install with `pip install python-jose`.")
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return int(payload["sub"])  # raises if missing/invalid
