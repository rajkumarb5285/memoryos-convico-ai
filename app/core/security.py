import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from app.core.config import settings
from app.core.exceptions import AuthenticationError, ForbiddenError


@dataclass
class Principal:
    user_id: str
    username: str | None = None
    is_authenticated: bool = True


def hash_password(password: str) -> str:
    """Generate a salted SHA-256 password hash."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """Verify a plain password against a salted SHA-256 hash."""
    try:
        salt, hashed = stored_hash.split(":")
        expected = hashlib.sha256((salt + plain_password).encode("utf-8")).hexdigest()
        return hmac.compare_digest(expected, hashed)
    except Exception:
        return False


def create_access_token(
    user_id: str,
    username: str | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT access token for a user."""
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode: dict[str, Any] = {
        "sub": str(user_id),
        "username": username or str(user_id),
        "exp": expire,
        "iat": datetime.now(UTC),
    }
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Principal:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
        user_id: str = payload.get("sub")
        if not user_id:
            raise AuthenticationError("Token missing subject claim")
        return Principal(user_id=user_id, username=payload.get("username"))
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Access token has expired")
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid access token: {e!s}")


def verify_user_ownership(principal: Principal, target_user_id: str) -> None:
    """Enforce that the authenticated principal matches the target user."""
    if str(principal.user_id) != str(target_user_id):
        raise ForbiddenError("Access forbidden: authenticated principal does not match resource owner")
