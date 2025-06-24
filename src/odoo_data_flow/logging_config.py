"""Centralized logging configuration for the odoo-data-flow application."""

import logging
import sys

# Get the root logger for the application package
log = logging.getLogger("odoo_data_flow")


def setup_logging(verbose=False):
    """Configures the root logger for the application.

    This function sets up a handler that prints logs to the console
    with a consistent format.

    Args:
        verbose (bool): If True, the logging level is set to DEBUG.
                        Otherwise, it's set to INFO.
    """
    # Determine the logging level
    level = logging.DEBUG if verbose else logging.INFO
    log.setLevel(level)

    # Clear any existing handlers to avoid duplicate logs if this is called multiple times
    if log.hasHandlers():
        log.handlers.clear()

    # Create a handler to print to the console
    handler = logging.StreamHandler(sys.stdout)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add the handler to the logger
    log.addHandler(handler)
