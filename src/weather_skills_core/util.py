"""Transient-error heuristics and env checks."""

import os
import re

from weather_skills_core.errors import UsageError

_STATUS_RE = re.compile(r"\b(?:429|500|502|503|504)\b")
_TIMEOUT_MARKERS = ("timed out", "timeout")
_CONNECTION_MARKERS = (
    "connection error",
    "connection reset",
    "connection refused",
    "connection aborted",
)


def is_transient(exc: Exception) -> bool:
    """True if exc text looks like a retryable HTTP/timeout/connection failure."""
    text = str(exc).lower()
    if _STATUS_RE.search(text):
        return True
    if any(marker in text for marker in _TIMEOUT_MARKERS):
        return True
    return any(marker in text for marker in _CONNECTION_MARKERS)


def require_env(*names: str, message: str | None = None) -> tuple:
    """Return env values in order; raise UsageError if any are missing/empty."""
    values = [os.environ.get(name) for name in names]
    missing = [name for name, value in zip(names, values, strict=True) if not value]
    if missing:
        raise UsageError(message or f"missing required env var(s): {', '.join(missing)}")
    return tuple(values)
