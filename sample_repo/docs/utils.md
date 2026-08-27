# Utilities

Small helper functions shared across the codebase.

## Email Validation

`validate_email(email)` performs a lightweight regex check
(`^[^@\s]+@[^@\s]+\.[^@\s]+$`). It is intended for form validation, not
RFC 5322-compliant parsing or deliverability checking.

## IDs and Timestamps

`generate_uuid()` returns a random UUID4 string, used as the primary key
for new records. `format_timestamp(epoch_seconds, fmt)` converts a Unix
epoch timestamp into a human-readable UTC string using a strftime-style
format string, defaulting to `"%Y-%m-%d %H:%M:%S UTC"`.

## Retry Decorator

`retry(max_attempts, delay_seconds, backoff)` is a decorator factory that
wraps a function in retry-with-exponential-backoff logic: on exception, it
waits `delay_seconds`, then multiplies the delay by `backoff` before each
subsequent attempt, up to `max_attempts` total tries. If every attempt
fails, the last exception raised is re-raised to the caller.
