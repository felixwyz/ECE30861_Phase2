"""Centralized logging utilities for the ece_30801 project."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

# Name shared by all loggers created through this utility
_LOGGER_NAME = "ece_30801"
_LOGGER = logging.getLogger(_LOGGER_NAME)
_CONFIGURED = False

_LEVEL_MAP = {
    1: logging.INFO,
    2: logging.DEBUG,
}


def _parse_level(raw_level: Optional[str]) -> int:
    """Convert the LOG_LEVEL string to a supported numeric level."""
    if raw_level is None:
        return 0
    try:
        value = int(raw_level)
    except (TypeError, ValueError):
        return 0
    return value if value in (0, 1, 2) else 0


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    _CONFIGURED = True

    log_file = os.getenv("LOG_FILE")
    if not log_file:
        # Nothing to configure; leave logger disabled.
        _LOGGER.disabled = True
        return

    if not log_file.endswith(".log"):
        # Defer hard validation to the CLI layer; keep logger disabled here.
        _LOGGER.disabled = True
        return

    level = _parse_level(os.getenv("LOG_LEVEL"))

    log_path = Path(log_file).expanduser()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fall back to disabled logging if we cannot create directories.
        _LOGGER.disabled = True
        return

    # Always start from a clean handler state.
    _LOGGER.handlers.clear()

    if level <= 0:
        # Truncate/touch the log file but keep logging disabled.
        try:
            log_path.write_text("", encoding="utf-8")
        except Exception:
            # Ignore failures to truncate; best effort.
            pass
        _LOGGER.disabled = True
        return

    # Set up file handler that writes UTF-8 logs.
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(_LEVEL_MAP.get(level, logging.INFO))
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Configure the shared root logger for this project.
    _LOGGER.setLevel(logging.DEBUG)
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = False
    _LOGGER.disabled = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger, scoped under the project namespace."""
    _configure_root_logger()
    if not name:
        return _LOGGER
    return _LOGGER.getChild(name)
