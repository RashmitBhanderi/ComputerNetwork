"""Shared logging setup — console + rotating file handler under logs/."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config import LOG_DIR

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once. Idempotent across imports."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        LOG_DIR / "trading_bot.log", maxBytes=5_000_000, backupCount=5
    )
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)

    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "apscheduler", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(name)
