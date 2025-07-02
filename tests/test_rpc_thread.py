"""Test the RpcThread class."""

from unittest.mock import MagicMock, patch

import pytest

from odoo_data_flow.lib.internal.rpc_thread import RpcThread


def test_rpc_thread_invalid_max_connection() -> None:
    """Test invalid max connection.

    Tests that initializing RpcThread with a non-positive max_connection
    raises a ValueError.
    """
    with pytest.raises(ValueError, match="max_connection must be a positive integer"):
        RpcThread(0)

    with pytest.raises(ValueError, match="max_connection must be a positive integer"):
        RpcThread(-1)


@patch("odoo_data_flow.lib.internal.rpc_thread.log.error")
def test_rpc_thread_wait_handles_exception(mock_log_error: MagicMock) -> None:
    """Test Wait handle exception.

    Tests that the wait() method correctly catches and logs exceptions
    from worker threads.
    """
    # 1. Setup
    rpc_thread = RpcThread(max_connection=1)

    def failing_function() -> None:
        """A simple function that always raises an error."""
        raise ValueError("This is a test failure.")

    # 2. Action
    # Spawn a thread that will execute the failing function
    rpc_thread.spawn_thread(failing_function, args=[], kwargs={})
    # The wait method should catch the exception and log it
    rpc_thread.wait()

    # 3. Assertions
    mock_log_error.assert_called_once()
    # Check that the log message contains the exception's message
    log_message = mock_log_error.call_args[0][0]
    assert "A task in a worker thread failed" in log_message
    assert "This is a test failure" in log_message


def test_rpc_thread_thread_number() -> None:
    """Test the thread number.

    Tests that the thread_number() method returns the correct count of
    submitted tasks.
    """
    # 1. Setup
    rpc_thread = RpcThread(max_connection=2)

    def dummy_function() -> None:
        pass

    # 2. Action
    rpc_thread.spawn_thread(dummy_function, [], {})
    rpc_thread.spawn_thread(dummy_function, [], {})
    rpc_thread.spawn_thread(dummy_function, [], {})

    # 3. Assertions
    assert rpc_thread.thread_number() == 3

    # Clean up the threads
    rpc_thread.wait()
