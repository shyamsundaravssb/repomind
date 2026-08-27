"""
auth.py — Authentication and authorization utilities.

Handles password hashing, JWT-based access tokens, and user
authentication including a login-attempt lockout policy.
"""

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field


# In-memory store of failed login attempts, keyed by username.
# In a real system this would live in Redis or the DB, not process memory.
_FAILED_ATTEMPTS: dict[str, list[float]] = {}

MAX_ATTEMPTS = 5
LOCKOUT_WINDOW_SECONDS = 300  # 5 minutes


@dataclass
class AuthResult:
    """Result of an authentication attempt."""
    success: bool
    reason: str | None = None
    token: str | None = None
    locked_until: float | None = None


def hash_password(password: str, salt: bytes | None = None) -> str:
    """
    Hash a plaintext password using PBKDF2-HMAC-SHA256.

    Args:
        password: The plaintext password to hash.
        salt: Optional salt bytes. A new random salt is generated if omitted.

    Returns:
        A string of the form "<salt_hex>:<hash_hex>".
    """
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Verify a plaintext password against a stored salt:hash string.

    Args:
        password: The plaintext password supplied by the user.
        stored_hash: The "<salt_hex>:<hash_hex>" string from the database.

    Returns:
        True if the password matches, False otherwise.
    """
    salt_hex, hash_hex = stored_hash.split(":")
    salt = bytes.fromhex(salt_hex)
    candidate = hash_password(password, salt)
    _, candidate_hash_hex = candidate.split(":")
    return hmac.compare_digest(candidate_hash_hex, hash_hex)


def create_access_token(username: str, ttl_seconds: int = 3600) -> str:
    """
    Create a signed, opaque access token for a username.

    This is a simplified stand-in for a real JWT — it concatenates the
    username and expiry with a random nonce and hashes it, rather than
    implementing full JWT signing.

    Args:
        username: The authenticated user's username.
        ttl_seconds: How long the token should remain valid, in seconds.

    Returns:
        An opaque token string.
    """
    expiry = time.time() + ttl_seconds
    nonce = secrets.token_hex(8)
    payload = f"{username}:{expiry}:{nonce}"
    signature = hashlib.sha256(payload.encode()).hexdigest()
    return f"{payload}:{signature}"


def decode_access_token(token: str) -> tuple[str, float] | None:
    """
    Decode and validate an access token produced by create_access_token.

    Args:
        token: The token string to validate.

    Returns:
        A tuple of (username, expiry_timestamp) if the token is valid and
        unexpired, otherwise None.
    """
    try:
        username, expiry_str, nonce, signature = token.split(":")
    except ValueError:
        return None

    payload = f"{username}:{expiry_str}:{nonce}"
    expected_signature = hashlib.sha256(payload.encode()).hexdigest()
    if not hmac.compare_digest(expected_signature, signature):
        return None

    expiry = float(expiry_str)
    if expiry < time.time():
        return None

    return username, expiry


def authenticate_user(
    username: str,
    password: str,
    stored_hash: str,
    now: float | None = None,
) -> AuthResult:
    """
    Authenticate a user, enforcing a lockout policy after repeated failures.

    This function is intentionally long and does several things in sequence:
    checks for an active lockout, verifies the password, records failures,
    triggers lockout when the failure threshold is crossed, and on success
    clears the failure history and issues an access token. It exists in this
    unified form (rather than split into five tiny functions) to mirror how
    auth flows are often written in real small-to-mid-size codebases, and to
    give the AST-aware chunker something meaningfully large to keep intact.

    Args:
        username: The username attempting to log in.
        password: The plaintext password supplied.
        stored_hash: The stored "<salt_hex>:<hash_hex>" for this user.
        now: Injectable current time (epoch seconds), for testability.
             Defaults to time.time().

    Returns:
        An AuthResult describing success/failure, and a token on success.
    """
    if now is None:
        now = time.time()

    attempts = _FAILED_ATTEMPTS.get(username, [])
    # Drop attempts outside the lockout window before evaluating anything.
    recent_attempts = [t for t in attempts if now - t < LOCKOUT_WINDOW_SECONDS]
    _FAILED_ATTEMPTS[username] = recent_attempts

    if len(recent_attempts) >= MAX_ATTEMPTS:
        locked_until = recent_attempts[0] + LOCKOUT_WINDOW_SECONDS
        return AuthResult(
            success=False,
            reason="account_locked",
            locked_until=locked_until,
        )

    if not verify_password(password, stored_hash):
        recent_attempts.append(now)
        _FAILED_ATTEMPTS[username] = recent_attempts

        if len(recent_attempts) >= MAX_ATTEMPTS:
            locked_until = recent_attempts[0] + LOCKOUT_WINDOW_SECONDS
            return AuthResult(
                success=False,
                reason="account_locked",
                locked_until=locked_until,
            )

        remaining = MAX_ATTEMPTS - len(recent_attempts)
        return AuthResult(
            success=False,
            reason=f"invalid_credentials_{remaining}_attempts_remaining",
        )

    # Success: clear failure history and issue a token.
    _FAILED_ATTEMPTS.pop(username, None)
    token = create_access_token(username)
    return AuthResult(success=True, token=token)


def logout(token: str) -> bool:
    """
    Log out a user by invalidating their token.

    Note: since tokens here are stateless (self-contained, not stored
    server-side), this is a no-op placeholder that always returns True.
    A real implementation would add the token to a revocation list.

    Args:
        token: The token to invalidate.

    Returns:
        True (always, in this simplified implementation).
    """
    return True