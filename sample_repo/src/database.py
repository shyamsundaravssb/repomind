"""
database.py — A minimal in-memory "database" layer with a fake connection
pool, basic CRUD operations, and a schema migration runner.

This stands in for a real SQL/ORM layer so the sample repo has no external
dependencies (sqlite3, psycopg2, etc.) to install.
"""

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class User:
    """A user record."""
    id: str
    username: str
    email: str
    password_hash: str
    created_at: float = field(default_factory=time.time)
    is_active: bool = True


class ConnectionPool:
    """
    A fake connection pool that simulates checkout/checkin semantics
    without opening real network connections.

    Real pools (e.g. SQLAlchemy's QueuePool) track in-use vs idle
    connections and block or raise when exhausted; this mimics that
    shape at a small scale for demonstration purposes.
    """

    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self._in_use = 0

    def checkout(self) -> "Connection":
        """Acquire a connection from the pool, raising if exhausted."""
        if self._in_use >= self.max_connections:
            raise RuntimeError("connection pool exhausted")
        self._in_use += 1
        return Connection(self)

    def checkin(self) -> None:
        """Release a connection back to the pool."""
        self._in_use = max(0, self._in_use - 1)


class Connection:
    """A handle representing a checked-out connection."""

    def __init__(self, pool: ConnectionPool):
        self._pool = pool
        self.closed = False

    def close(self) -> None:
        """Close the connection, returning it to the pool."""
        if not self.closed:
            self._pool.checkin()
            self.closed = True


# The "table" itself: an in-memory dict keyed by user id.
_USERS: dict[str, User] = {}
_USERNAME_INDEX: dict[str, str] = {}  # username -> id


def create_user(username: str, email: str, password_hash: str) -> User:
    """
    Insert a new user record.

    Args:
        username: Unique username for the new user.
        email: The user's email address.
        password_hash: Pre-hashed password (see auth.hash_password).

    Returns:
        The newly created User.

    Raises:
        ValueError: If the username is already taken.
    """
    if username in _USERNAME_INDEX:
        raise ValueError(f"username already exists: {username}")

    user_id = str(uuid.uuid4())
    user = User(
        id=user_id,
        username=username,
        email=email,
        password_hash=password_hash,
    )
    _USERS[user_id] = user
    _USERNAME_INDEX[username] = user_id
    return user


def get_user_by_id(user_id: str) -> User | None:
    """Fetch a user by primary key, or None if not found."""
    return _USERS.get(user_id)


def get_user_by_username(username: str) -> User | None:
    """Fetch a user by username, or None if not found."""
    user_id = _USERNAME_INDEX.get(username)
    if user_id is None:
        return None
    return _USERS.get(user_id)


def update_user_email(user_id: str, new_email: str) -> User | None:
    """
    Update a user's email address.

    Args:
        user_id: The id of the user to update.
        new_email: The new email address.

    Returns:
        The updated User, or None if no user with that id exists.
    """
    user = _USERS.get(user_id)
    if user is None:
        return None
    user.email = new_email
    return user


def deactivate_user(user_id: str) -> bool:
    """
    Mark a user as inactive (soft delete).

    Args:
        user_id: The id of the user to deactivate.

    Returns:
        True if a user was found and deactivated, False otherwise.
    """
    user = _USERS.get(user_id)
    if user is None:
        return False
    user.is_active = False
    return True


def run_migration(target_version: int, current_version: int = 0) -> list[str]:
    """
    Simulate running schema migrations from current_version to target_version.

    This is a deliberately longer function covering several migration
    "steps" in one place, again to give the code chunker a nontrivial
    unit to preserve. Each step is a no-op here (no real schema to
    migrate against an in-memory dict) but logs what it would have done.

    Args:
        target_version: The schema version to migrate to.
        current_version: The schema version to migrate from. Defaults to 0.

    Returns:
        A list of human-readable log lines describing each applied step.

    Raises:
        ValueError: If target_version is less than current_version.
    """
    if target_version < current_version:
        raise ValueError(
            f"cannot migrate backwards: {current_version} -> {target_version}"
        )

    log: list[str] = []
    version = current_version

    while version < target_version:
        next_version = version + 1
        if next_version == 1:
            log.append("v1: create users table")
        elif next_version == 2:
            log.append("v2: add is_active column to users")
        elif next_version == 3:
            log.append("v3: add created_at column to users")
        elif next_version == 4:
            log.append("v4: add unique index on username")
        else:
            log.append(f"v{next_version}: no-op (unrecognized version step)")
        version = next_version

    log.append(f"migration complete: now at v{target_version}")
    return log