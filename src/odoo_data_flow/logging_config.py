"""Centralized logging configuration for the odoo-data-flow application."""

import logging
import sys
from typing import Optional

# Get the root logger for the application package
log = logging.getLogger("odoo_data_flow")


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Configures the root logger for the application.

    This function sets up handlers to print logs to the console and optionally
    to a specified file.

    Args:
        verbose: If True, the logging level is set to DEBUG.
                 Otherwise, it's set to INFO.
        log_file: If provided, logs will also be written to this file path.
    """
    # Determine the logging level
    level = logging.DEBUG if verbose else logging.INFO
    log.setLevel(level)

    # Clear any existing handlers to avoid duplicate logs if this is called
    # multiple times
    if log.hasHandlers():
        log.handlers.clear()

    # Create a formatter to be used by all handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Always create a handler to print to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)

    # If a log file is specified, create a file handler as well
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            log.addHandler(file_handler)
            log.info(f"Logging to file: {log_file}")
        except Exception as e:
            log.error(f"Failed to set up log file at {log_file}: {e}")
