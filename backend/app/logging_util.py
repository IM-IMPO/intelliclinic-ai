"""Minimal structured logging.

The project has no logging dependency and needs none - stdlib ``logging`` with a
single named logger is enough. It exists because every ``except`` that returns a
fallback value is otherwise invisible: a corrupt document registry looks exactly
like an empty one, and a lost idempotency record looks exactly like a first
submission. Without a logger those two cases are indistinguishable from normal
operation, which is precisely the "silent failure" this codebase refuses to
ship.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False

# One logger per module, all under this parent, so LOG_LEVEL controls the whole
# app with a single variable and no per-module configuration.
LOGGER_NAME = "intelliclinic"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the app logger, configuring it on first use."""

    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        _CONFIGURED = True
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)
