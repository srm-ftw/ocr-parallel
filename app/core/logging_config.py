"""Logging configuration."""

import logging
import sys
from typing import Optional

from app.config import settings


def setup_logging(log_level: Optional[str] = None):
    """
    Configure logging for the application.

    Args:
        log_level: Log level (defaults to settings.LOG_LEVEL)
    """
    level = log_level or settings.log_level
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    numeric_level = log_level_map.get(level.upper(), logging.INFO)

    # Choose format based on environment
    if settings.is_production:
        # Production format: more structured, less verbose
        log_format = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
    else:
        # Dev format: more detailed
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,  # Override any existing configuration
    )

    # Set log levels for third-party libraries (reduce noise)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # In production, reduce verbosity of uvicorn access logs
    if settings.is_production:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured - Level: {level}, Environment: {settings.environment}"
    )

