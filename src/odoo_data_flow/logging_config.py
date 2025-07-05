"""Centralized logging configuration for the odoo-data-flow application."""

import logging
from typing import Optional

from rich.logging import RichHandler

# Get the root logger for the application package
log = logging.getLogger("odoo_data_flow")


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Configures the root logger for the application.

    This function sets up handlers to print logs to the console and optionally
    to a specified file. It uses the 'rich' library to provide colorful,
    easy-to-read console output.

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

    # Create a rich handler for beautiful, colorful console output
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        log_time_format="[%X]",
    )
    log.addHandler(console_handler)

    # If a log file is specified, create a standard file handler as well.
    # We use a standard handler here to ensure the log file contains plain text
    # without any color codes.
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode="a")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            log.addHandler(file_handler)
            log.info(f"Logging to file: [bold cyan]{log_file}[/bold cyan]")
        except Exception as e:
            log.error(f"Failed to set up log file at {log_file}: {e}")
