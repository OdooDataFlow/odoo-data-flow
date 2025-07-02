"""Test the centralized logging configuration."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    # 2. Action: Call setup_logging, which should clear the existing handler
    setup_logging()

    # 3. Assertions
    # We should only have the new console handler, not the old NullHandler
    assert len(log.handlers) == 1
    assert isinstance(log.handlers[0], logging.StreamHandler)


def test_setup_logging_clears_existing_handlers() -> None:
    """Test logging clear exsisting handlers.

    Tests that calling setup_logging multiple times does not add duplicate handlers.
    This covers the `if log.hasHandlers():` branch.
    """
    # 1. Setup
    log.handlers.clear()
    # Add a dummy handler to start with
    log.addHandler(logging.NullHandler())

    # 2. Action: Call setup_logging, which should clear the existing handler
    setup_logging()

    # 3. Assertions
    # We should only have the new console handler, not the old NullHandler
    assert len(log.handlers) == 1
    assert isinstance(log.handlers[0], logging.StreamHandler)


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


@patch("odoo_data_flow.logging_config.logging.FileHandler")
@patch("odoo_data_flow.logging_config.log.error")
def test_setup_logging_file_creation_error(
    mock_log_error: MagicMock, mock_file_handler: MagicMock
) -> None:
    """Test Logging creation error.

    Tests that an error is logged if the FileHandler cannot be created.
    This covers the `except Exception` block.
    """
    # 1. Setup
    log.handlers.clear()
    # Configure the mock to raise an error when instantiated
    mock_file_handler.side_effect = OSError("Permission denied")

    # 2. Action
    setup_logging(log_file="unwritable/path/test.log")

    # 3. Assertions
    # The console handler should still be added
    assert len(log.handlers) == 1
    # The error should have been logged
    mock_log_error.assert_called_once()
    assert "Failed to set up log file" in mock_log_error.call_args[0][0]
