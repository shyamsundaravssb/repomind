# Database Layer

This document describes the in-memory data layer: the connection pool,
the `User` model, CRUD operations, and schema migrations.

## Connection Pool

`ConnectionPool` simulates checkout/checkin semantics similar to a real
pool like SQLAlchemy's `QueuePool`. `checkout()` raises `RuntimeError` if
`max_connections` is already in use; `checkin()` (called via `Connection.close()`)
returns a slot to the pool. This is an in-memory simulation with no real
network connections involved.

## User Model

`User` is a dataclass with `id`, `username`, `email`, `password_hash`,
`created_at` (defaults to creation time), and `is_active` (defaults to `True`).

## CRUD Operations

- `create_user(username, email, password_hash)` inserts a new user, raising
  `ValueError` if the username is already taken.
- `get_user_by_id(user_id)` / `get_user_by_username(username)` fetch a user,
  returning `None` if not found.
- `update_user_email(user_id, new_email)` updates the email on an existing
  user record.
- `deactivate_user(user_id)` performs a soft delete by setting `is_active`
  to `False` rather than removing the record.

## Migrations

`run_migration(target_version, current_version)` walks forward through
schema versions 1–4 (create users table, add `is_active`, add `created_at`,
add unique index on username), logging each step. It raises `ValueError`
if asked to migrate backwards. Versions beyond 4 are treated as no-ops.
