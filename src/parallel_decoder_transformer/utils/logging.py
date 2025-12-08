"""Logging setup tailored for multi-stream experiments."""

from __future__ import annotations

import logging
from logging import Logger
from typing import Iterable, Optional


def configure_logging(
    level: int = logging.INFO,
    *,
    name: str = "parallel decoder transformer",
    propagate: bool = False,
    extra_loggers: Optional[Iterable[str]] = None,
) -> Logger:
    """Configure and return a logger instance, optionally binding child loggers.

    Parameters
    ----------
    level: int
        Logging verbosity.
    name: str
        Logical logger namespace.
    """

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    handler.setFormatter(formatter)

    def _attach(target: Logger) -> None:
        if not any(isinstance(existing, logging.StreamHandler) for existing in target.handlers):
            target.addHandler(handler)
        target.setLevel(level)
        target.propagate = propagate

    logger = logging.getLogger(name)
    _attach(logger)

    if extra_loggers:
        for logger_name in extra_loggers:
            _attach(logging.getLogger(logger_name))

    return logger
