"""Logging configuration for the serving application."""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a structured format suitable for stdout collection.

    Args:
        level: Logging level for the root logger. Defaults to INFO.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)
