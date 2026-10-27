"""Load persisted JSON state, validating shape as well as syntax.

A file that parses but has the wrong shape is exactly as unusable as one that
does not parse, and both are just as dangerous as an empty file: an unreadable
idempotency store looks like a first submission, and an unreadable document
registry looks like an empty knowledge base. Neither failure is visible to the
operator unless it is raised, so the shape is checked here rather than left to
be discovered later as a TypeError on some unrelated line.

Callers supply their own exception factory so each one keeps its own error type
and therefore its own handling contract at the call site.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .logging_util import get_logger


def read_json_state(
    path: Path,
    *,
    is_valid: Callable[[Any], bool],
    on_error: Callable[[Path, str], Exception],
    logger_name: str,
    what: str,
) -> Any:
    """Return the parsed JSON at *path*, or raise rather than guess.

    *is_valid* decides whether the parsed value has the shape this caller needs.
    *on_error* builds the caller's own exception from the path and a short
    description of what was wrong.
    """

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        get_logger(logger_name).error(
            "%s %s could not be read (%s: %s)", what, path, type(exc).__name__, exc
        )
        raise on_error(path, f"{type(exc).__name__}: {exc}") from exc

    try:
        value = json.loads(raw)
    except Exception as exc:
        get_logger(logger_name).error(
            "%s %s is not valid JSON (%s: %s)", what, path, type(exc).__name__, exc
        )
        raise on_error(path, f"{type(exc).__name__}: {exc}") from exc

    if not is_valid(value):
        detail = f"expected {what}, got {type(value).__name__}"
        get_logger(logger_name).error("%s %s has the wrong shape: %s", what, path, detail)
        raise on_error(path, detail)

    return value
