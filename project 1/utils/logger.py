"""Structured and colored logging utility using Rich and JSON Lines output.

This module provides logging capabilities that output colored formatting to
the standard console via Rich, while simultaneously writing structured JSON lines
to timestamped session log files for persistent debugging and auditing.
"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler

from config import get_config


class JSONLinesFormatter(logging.Formatter):
    """Custom logging formatter that encodes log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record into a single-line JSON string.

        Args:
            record: The logging.LogRecord instance to format.

        Returns:
            str: JSON-encoded line representing the record.
        """
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include session metadata or extra fields if present
        if hasattr(record, "session_id"):
            log_payload["session_id"] = record.session_id
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_payload["extra"] = record.extra_data

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload, default=str)


# Global console instance
_console: Console = Console()
_active_loggers: Dict[str, logging.Logger] = {}


def get_logger(name: str = "MultimodalAttendance", session_id: Optional[str] = None) -> logging.Logger:
    """Retrieve or create a logger configured with Rich console and optional file output.

    Args:
        name: Name of the logger or module.
        session_id: Optional session identifier for organizing session log files.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger_key = f"{name}_{session_id or 'default'}"
    if logger_key in _active_loggers:
        return _active_loggers[logger_key]

    logger = logging.getLogger(logger_key)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if logger was already created
    if not logger.handlers:
        # Rich Console Handler
        rich_handler = RichHandler(
            console=_console,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        rich_handler.setLevel(logging.INFO)
        rich_format = logging.Formatter("%(message)s")
        rich_handler.setFormatter(rich_format)
        logger.addHandler(rich_handler)

        # File Handler (JSON Lines)
        try:
            cfg = get_config()
            logs_dir = Path(cfg.storage.logs_dir)
            logs_dir.mkdir(parents=True, exist_ok=True)

            date_str = datetime.now().strftime("%Y%m%d")
            file_name = f"session_{session_id}_{date_str}.jsonl" if session_id else f"app_{date_str}.jsonl"
            file_path = logs_dir / file_name

            file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(JSONLinesFormatter())
            logger.addHandler(file_handler)
        except Exception as err:
            logger.warning(f"Could not attach file handler to logger: {err}")

    _active_loggers[logger_key] = logger
    return logger


def setup_session_logger(session_id: str) -> logging.Logger:
    """Convenience function to initialize and return a session-specific logger.

    Args:
        session_id: Unique identifier for the classroom session.

    Returns:
        logging.Logger: The configured session logger.
    """
    return get_logger("SessionManager", session_id=session_id)
