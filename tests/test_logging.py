"""Test Logging functionality."""

import logging
from pathlib import Path

from odoo_data_flow.logging_config import log, setup_logging


def test_setup_logging_console_only() -> None:
    """Tests that logging is set up correctly for console-only output."""
    # 1. Setup: Ensure logger is in a clean state
    log.handlers.clear()

    # 2. Action: Configure logging without a file path
    setup_logging(verbose=True)

    # 3. Assertions
    assert len(log.handlers) == 1, (
        "There should be exactly one handler for the console."
    )
    assert isinstance(log.handlers[0], logging.StreamHandler)
    assert not isinstance(log.handlers[0], logging.FileHandler)


def test_setup_logging_with_file(tmp_path: Path) -> None:
    """Test log file writing.

    Tests that logging is set up with both console and file handlers
    when a log file path is provided.
    """
    # 1. Setup
    log.handlers.clear()
    log_file = tmp_path / "test.log"

    # 2. Action
    setup_logging(verbose=True, log_file=str(log_file))

    # 3. Assertions
    assert len(log.handlers) == 2, "There should be two handlers: console and file."

    # Check that we have one of each type of handler
    handler_types = [type(h) for h in log.handlers]
    assert logging.StreamHandler in handler_types
    assert logging.FileHandler in handler_types

    # Find the file handler and check its path
    file_handler = next(
        (h for h in log.handlers if isinstance(h, logging.FileHandler)), None
    )
    assert file_handler is not None
    assert file_handler.baseFilename == str(log_file)


def test_log_output_is_written_to_file(tmp_path: Path) -> None:
    """Tests that log messages are correctly written to the specified log file."""
    # 1. Setup
    log.handlers.clear()
    log_file = tmp_path / "output.log"
    test_message = "This is a test message for the log file."

    # 2. Action
    setup_logging(verbose=False, log_file=str(log_file))
    log.info(test_message)

    # To ensure the log is written, we need to shut down the logging system
    # This closes the file handle.
    logging.shutdown()

    # 3. Assertions
    assert log_file.exists(), "Log file was not created."
    log_content = log_file.read_text()
    assert test_message in log_content
