# Authentication

This document describes the authentication flow used across the service:
password handling, access tokens, login lockout policy, and password reset.

## Password Hashing

Passwords are never stored in plaintext. `hash_password()` uses
PBKDF2-HMAC-SHA256 with a random 16-byte salt and 100,000 iterations,
returning a string of the form `<salt_hex>:<hash_hex>`. `verify_password()`
re-derives the hash from a candidate password and the stored salt, then
compares digests using a constant-time comparison to avoid timing attacks.

## Access Tokens

`create_access_token()` issues an opaque token binding a username to an
expiry timestamp, signed with a SHA-256 digest over the payload. Tokens are
stateless: no server-side session store is required to validate them.
`decode_access_token()` reverses this — it recomputes the expected signature,
compares it against the token's signature, and checks that the token has not
expired, returning `(username, expiry)` on success or `None` otherwise.

## Login Lockout Policy

`authenticate_user()` enforces a lockout policy to slow down brute-force
attempts: after `MAX_ATTEMPTS` (5) failed logins within a
`LOCKOUT_WINDOW_SECONDS` (300 second / 5 minute) window, the account is
locked and further attempts are rejected with an `account_locked` reason
until the window elapses. A successful login clears the failure history for
that user and issues a fresh access token.

## Password Reset

Users who forget their password can request a reset via
`request_password_reset(username)`, which generates a single-use, 15-minute
reset token and emails it to the address on file. The user submits the token
along with a new password to `complete_password_reset(token, new_password)`,
which validates the token, updates the stored password hash, and invalidates
any outstanding access tokens for that account. Reset tokens are rate-limited
to 3 requests per hour per account to prevent email-bombing abuse.

## Logout

`logout(token)` invalidates a token. In the current stateless-token design
this is a placeholder — a production version would add the token to a
revocation list checked during `decode_access_token()`.
