"""
IQAS Logging Module
===================
Structured logging with loguru — file + console handlers.
"""

import sys
from loguru import logger
from utils.config import LOG_FILE, LOG_LEVEL

# Remove default handler
logger.remove()

# Console handler — colorized, concise
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# File handler — detailed, rotated
logger.add(
    str(LOG_FILE),
    level="DEBUG",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{module}:{function}:{line} | "
        "{message}"
    ),
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    enqueue=True,  # Thread-safe
)


def get_logger(name: str = "iqas"):
    """Return a contextualized logger instance."""
    return logger.bind(module=name)
