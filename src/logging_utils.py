"""Structured logging configuration shared by every entry point.

Logs carry a timestamp, level and module name so a long generation run leaves
a readable trace of which stage was slow and where an example failed.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

#: Format used for console logs.
_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"

#: Timestamp format, seconds resolution.
_DATE_FORMAT: Final[str] = "%H:%M:%S"

#: Third-party loggers that are noisy at INFO level.
_QUIET_LOGGERS: Final[tuple[str, ...]] = (
    "urllib3",
    "filelock",
    "fsspec",
    "matplotlib",
    "matplotlib.font_manager",
    "datasets",
    "huggingface_hub",
)


def configure_logging(level: int = logging.INFO) -> None:
    """Install a console log handler and quieten noisy dependencies.

    Args:
        level: Log level applied to this project's loggers.
    """
    root = logging.getLogger()
    root.setLevel(level)

    for existing in list(root.handlers):
        root.removeHandler(existing)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
