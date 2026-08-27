"""
utils.py — Small, generic helper functions used across the codebase:
email validation, timestamp formatting, ID generation, and a retry decorator.
"""

import functools
import re
import time
import uuid
from datetime import datetime, timezone


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_email(email: str) -> bool:
    """
    Check whether a string looks like a valid email address.

    This is a lightweight regex check, not RFC 5322-compliant validation —
    sufficient for form validation, not for guaranteeing deliverability.

    Args:
        email: The string to validate.

    Returns:
        True if the string matches a basic email pattern, False otherwise.
    """
    return bool(_EMAIL_RE.match(email))


def generate_uuid() -> str:
    """Generate a random UUID4 string, used for user/record ids."""
    return str(uuid.uuid4())


def format_timestamp(epoch_seconds: float, fmt: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """
    Format a Unix epoch timestamp as a human-readable UTC string.

    Args:
        epoch_seconds: Seconds since the Unix epoch.
        fmt: A strftime-compatible format string.

    Returns:
        The formatted timestamp string.
    """
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return dt.strftime(fmt)


def retry(max_attempts: int = 3, delay_seconds: float = 0.5, backoff: float = 2.0):
    """
    Decorator that retries a function call on exception, with exponential
    backoff between attempts.

    Args:
        max_attempts: Maximum number of attempts before giving up.
        delay_seconds: Initial delay between attempts, in seconds.
        backoff: Multiplier applied to the delay after each failed attempt.

    Returns:
        A decorator that wraps the target function with retry logic.

    Raises:
        The last exception raised by the wrapped function, if all
        attempts are exhausted.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay_seconds
            last_exception = None

            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - intentional broad catch
                    last_exception = exc
                    attempt += 1
                    if attempt < max_attempts:
                        time.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        return wrapper

    return decorator