"""Test the low-level, multi-threaded import logic."""

import sys
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, mock_open, patch

import pytest
import requests  # type: ignore[import-untyped]
from rich.progress import Progress, TaskID

from odoo_data_flow.import_threaded import (
    RPCThreadImport,
    _create_batches,
    _filter_ignored_columns,
    _read_data_file,
    _setup_fail_file,
    import_data,
)


class TestRPCThreadImport:
    """Tests for the RPCThreadImport class."""

    def test_handle_odoo_messages_with_error_reason(self) -> None:
        """Tests that when add_error_reason is True, the reason is appended."""
        header = ["id", "name"]
        lines = [["1", "A"], ["2", "B"]]
        mock_writer = MagicMock()
        rpc_thread = RPCThreadImport(
            1, None, header, mock_writer, add_error_reason=True
        )
        messages = [{"message": "Generic Error"}]
        failed_lines = rpc_thread._handle_odoo_messages(messages, lines)
        assert failed_lines[0][-1] == "Generic Error\n".replace("\n", " | ")

    def test_handle_odoo_messages_no_error_reason(self) -> None:
        """Tests that when add_error_reason is False, the reason is not appended."""
        header = ["id", "name"]
        lines = [["1", "A"], ["2", "B"]]
        mock_writer = MagicMock()
        rpc_thread = RPCThreadImport(
            1, None, header, mock_writer, add_error_reason=False
        )
        messages = [{"message": "Generic Error", "record": 0}]
        failed_lines = rpc_thread._handle_odoo_messages(messages, lines)
        assert len(failed_lines[0]) == 2

    def test_handle_record_mismatch(self) -> None:
        """Tests the logic for handling a record count mismatch."""
        header = ["id", "name"]
        lines = [["1", "A"], ["2", "B"]]
        mock_writer = MagicMock()
        rpc_thread = RPCThreadImport(
            1, None, header, mock_writer, add_error_reason=True
        )
        response = {"ids": [123]}
        failed_lines = rpc_thread._handle_record_mismatch(response, lines)
        assert len(failed_lines) == 2
        assert "Record count mismatch" in failed_lines[0][2]

    def test_handle_rpc_error(self) -> None:
        """Tests the logic for handling a general RPC exception."""
        header = ["id", "name"]
        lines = [["1", "A"], ["2", "B"]]
        mock_writer = MagicMock()
        rpc_thread = RPCThreadImport(
            1, None, header, mock_writer, add_error_reason=True
        )
        error = Exception("Connection Timed Out")
        failed_lines = rpc_thread._handle_rpc_error(error, lines)
        assert len(failed_lines) == 2
        assert failed_lines[0][2] == "Connection Timed Out"

    @patch("odoo_data_flow.lib.internal.rpc_thread.RpcThread.wait")
    def test_wait_fallback_without_progress(self, mock_super_wait: MagicMock) -> None:
        """Tests that wait() calls super().wait() if no progress bar is given."""
        rpc_thread = RPCThreadImport(1, None, [], None)
        rpc_thread.wait()
        mock_super_wait.assert_called_once()

    def test_wait_updates_progress_bar(self) -> None:
        """Tests that wait() updates the progress bar on task completion."""
        mock_progress = MagicMock(spec=Progress)
        mock_task_id = MagicMock(spec=TaskID)
        rpc_thread = RPCThreadImport(
            1, None, [], None, progress=mock_progress, task_id=mock_task_id
        )

        # Simulate a completed future
        future = MagicMock()
        future.result.return_value = 5  # Simulate a batch of 5 records
        rpc_thread.futures = [future]

        with patch("concurrent.futures.as_completed", return_value=[future]):
            rpc_thread.wait()

        mock_progress.update.assert_called_once_with(mock_task_id, advance=5)

    @patch("odoo_data_flow.lib.internal.rpc_thread.log.error")
    def test_wait_handles_and_logs_exception(self, mock_log_error: MagicMock) -> None:
        """Tests that wait() correctly handles exceptions from futures."""
        mock_progress = MagicMock(spec=Progress)
        mock_task_id = MagicMock(spec=TaskID)
        rpc_thread = RPCThreadImport(
            1, None, [], None, progress=mock_progress, task_id=mock_task_id
        )
        future = MagicMock()
        test_exception = ValueError("Worker thread failed")
        future.result.side_effect = test_exception
        rpc_thread.futures = [future]

        with patch("concurrent.futures.as_completed", return_value=[future]):
            rpc_thread.wait()

        mock_log_error.assert_called_once()
        assert "A task in a worker thread failed" in mock_log_error.call_args[0][0]
        assert mock_log_error.call_args.kwargs["exc_info"] is True


class TestHelperFunctions:
    """Tests for the standalone helper functions in the module."""

    def test_read_data_file_not_found(self) -> None:
        """Tests that _read_data_file returns empty lists for a non-existent file."""
        header, data = _read_data_file("non_existent_file.csv", ";", "utf-8", 0)
        assert header == []
        assert data == []

    @patch("odoo_data_flow.import_threaded.open")
    def test_read_data_file_generic_exception(self, mock_open: MagicMock) -> None:
        """Tests that _read_data_file handles generic exceptions during read."""
        mock_open.side_effect = Exception("A generic read error")
        header, data = _read_data_file("any_file.csv", ";", "utf-8", 0)
        assert header == []
        assert data == []

    def test_read_data_file_no_id_column(self, tmp_path: Path) -> None:
        """Tests that _read_data_file raises ValueError if 'id' column is missing."""
        source_file = tmp_path / "no_id.csv"
        source_file.write_text("name,value\nTest,100")
        with pytest.raises(
            ValueError,
            match="Source file must contain an 'id' column for external IDs.",
        ):
            _read_data_file(str(source_file), ",", "utf-8", 0)

    def test_create_batches_split_by_size(self) -> None:
        """Tests that batches are created by size when the group value is the same."""
        header = ["id", "group_id"]
        data = [
            ["1", "A"],
            ["2", "A"],
            ["3", "A"],
            ["4", "A"],
            ["5", "A"],
        ]
        # Batch size of 3 should create two batches for group A
        batches = list(_create_batches(data, "group_id", header, 3, False))
        assert len(batches) == 2
        assert len(batches[0][1]) == 3
        assert len(batches[1][1]) == 2


class TestImportData:
    """Tests for the main import_data orchestrator function."""

    def test_import_data_no_header_or_data(self) -> None:
        """Tests that import_data raises ValueError if no data is provided."""
        with pytest.raises(ValueError, match="Please provide either a data file"):
            import_data(config_file="dummy.conf", model="dummy.model")

    @patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_config")
    def test_import_data_connection_fails(self, mock_get_conn: MagicMock) -> None:
        """Tests that the function exits gracefully if the connection fails."""
        mock_get_conn.side_effect = Exception("Cannot connect")
        result = import_data(
            config_file="bad.conf",
            model="dummy.model",
            header=["id"],
            data=[["1"]],
        )
        assert result is False
        mock_get_conn.assert_called_once()

    @patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_config")
    @patch("odoo_data_flow.import_threaded.open")
    def test_import_data_fail_file_oserror(
        self, mock_open: MagicMock, mock_get_conn: MagicMock
    ) -> None:
        """Tests that the function handles an OSError when opening the fail file."""
        mock_get_conn.return_value = MagicMock()
        mock_open.side_effect = OSError("Permission denied")
        result = import_data(
            config_file="dummy.conf",
            model="dummy.model",
            header=["id"],
            data=[["1"]],
            fail_file="protected/fail.csv",
        )
        assert result is False
        mock_open.assert_called_once()

    @patch("odoo_data_flow.import_threaded.RPCThreadImport")
    def test_import_data_ignore_columns(self, mock_rpc_thread: MagicMock) -> None:
        """Tests that the 'ignore' parameter correctly filters columns."""
        header = ["id", "name", "field_to_ignore"]
        data = [["1", "A", "ignore_me"]]

        with patch(
            "odoo_data_flow.import_threaded.conf_lib.get_connection_from_config"
        ):
            import_data(
                config_file="dummy.conf",
                model="dummy.model",
                header=header,
                data=data,
                ignore=["field_to_ignore"],
            )

        # Assert that the header passed to RPCThreadImport was filtered
        init_args = mock_rpc_thread.call_args.args
        filtered_header = init_args[2]
        assert filtered_header == ["id", "name"]

    @patch("odoo_data_flow.import_threaded.RPCThreadImport")
    @patch("odoo_data_flow.import_threaded.Progress")
    def test_import_data_with_progress(
        self, mock_progress: MagicMock, mock_rpc_thread: MagicMock
    ) -> None:
        """Tests that import_data correctly uses the rich progress bar."""
        header = ["id", "name"]
        data = [["1", "A"]]

        with patch(
            "odoo_data_flow.import_threaded.conf_lib.get_connection_from_config"
        ):
            import_data(
                config_file="dummy.conf",
                model="dummy.model",
                header=header,
                data=data,
            )

        mock_progress.assert_called_once()
        mock_rpc_thread.assert_called_once()
        # Check that the progress object was passed to the thread handler
        assert "progress" in mock_rpc_thread.call_args.kwargs

    @patch("odoo_data_flow.import_threaded.RPCThreadImport")
    def test_import_data_returns_true_on_success(
        self, mock_rpc_thread: MagicMock
    ) -> None:
        """Tests that import_data returns True on a successful run."""
        mock_rpc_thread.return_value.abort_flag = False
        with patch(
            "odoo_data_flow.import_threaded.conf_lib.get_connection_from_config"
        ):
            result = import_data(
                config_file="dummy.conf",
                model="dummy.model",
                header=["id"],
                data=[["1"]],
            )
        assert result is True

    @patch("odoo_data_flow.import_threaded.RPCThreadImport")
    def test_import_data_aborts_on_connection_error(
        self, mock_rpc_thread_class: MagicMock
    ) -> None:
        """Tests that the import process aborts on a connection error."""
        mock_rpc_instance = mock_rpc_thread_class.return_value

        # Corrected: Add type hints to the inner helper function
        def launch_side_effect(*args: Any, **kwargs: Any) -> None:
            mock_rpc_instance.abort_flag = True

        mock_rpc_instance.launch_batch.side_effect = launch_side_effect

        with patch(
            "odoo_data_flow.import_threaded.conf_lib.get_connection_from_config"
        ):
            result = import_data(
                config_file="dummy.conf",
                model="dummy.model",
                header=["id"],
                data=[
                    ["1"],
                    ["2"],
                    ["3"],
                ],  # Multiple records to create batches
                batch_size=1,
            )

        assert result is False


@patch("sys.maxsize", 100)
@patch("csv.field_size_limit")
def test_csv_field_size_limit_overflow(
    mock_field_size_limit: MagicMock,
) -> None:
    """Test that csv.field_size_limit handles OverflowError."""
    mock_field_size_limit.side_effect = [OverflowError, None]
    # Reload the module to re-run the top-level code
    import importlib

    importlib.reload(sys.modules["odoo_data_flow.import_threaded"])
    mock_field_size_limit.assert_called_with(10)


def test_handle_odoo_messages_no_record_details() -> None:
    """Test _handle_odoo_messages when Odoo sends generic messages."""
    rpc_thread = RPCThreadImport(1, MagicMock(), [], add_error_reason=True)
    messages = [{"message": "Generic Error"}]
    original_lines = [["1", "Test"], ["2", "Test2"]]
    failed_lines = rpc_thread._handle_odoo_messages(messages, original_lines)
    assert failed_lines == [
        ["1", "Test", "Generic Error | "],
        ["2", "Test2", "Generic Error | "],
    ]


def test_handle_rpc_error_with_error_reason() -> None:
    """Test _handle_rpc_error when add_error_reason is True."""
    rpc_thread = RPCThreadImport(1, MagicMock(), [], add_error_reason=True)
    error = Exception("RPC Failed")
    lines = [["1", "Test"]]
    failed_lines = rpc_thread._handle_rpc_error(error, lines)
    assert failed_lines == [["1", "Test", "RPC Failed"]]


def test_handle_record_mismatch_with_error_reason() -> None:
    """Test _handle_record_mismatch when add_error_reason is True."""
    rpc_thread = RPCThreadImport(1, MagicMock(), [], add_error_reason=True)
    response = {"ids": [1]}
    lines = [["1", "Test"], ["2", "Test2"]]
    failed_lines = rpc_thread._handle_record_mismatch(response, lines)
    assert failed_lines[0][2].startswith("Record count mismatch")


@patch("odoo_data_flow.import_threaded.RPCThreadImport.spawn_thread")
def test_launch_batch_abort_flag(mock_spawn_thread: MagicMock) -> None:
    """Test launch_batch when abort_flag is True."""
    rpc_thread = RPCThreadImport(1, MagicMock(), [])
    rpc_thread.abort_flag = True
    rpc_thread.launch_batch([["1", "Test"]], 1)
    mock_spawn_thread.assert_not_called()


@patch("concurrent.futures.as_completed")
@patch("concurrent.futures.ThreadPoolExecutor")
def test_wait_abort_flag(
    mock_thread_pool_executor: MagicMock, mock_as_completed: MagicMock
) -> None:
    """Test wait when abort_flag is True."""
    mock_executor_instance = MagicMock()
    mock_thread_pool_executor.return_value = mock_executor_instance

    # Pass mocks for progress and task_id to ensure the main wait logic is used
    mock_progress = MagicMock()
    mock_task_id = MagicMock()
    rpc_thread = RPCThreadImport(
        1, MagicMock(), [], progress=mock_progress, task_id=mock_task_id
    )

    rpc_thread.abort_flag = True

    # Provide a future so the loop is entered
    mock_future = MagicMock()
    mock_future.result.return_value = 1  # Simulate a processed line
    mock_as_completed.return_value = [mock_future]
    rpc_thread.futures = [mock_future]

    rpc_thread.wait()

    # Assert that the shutdown method of the mock executor instance was called correctly
    mock_executor_instance.shutdown.assert_called_once_with(
        wait=True, cancel_futures=True
    )


@patch("concurrent.futures.as_completed")
def test_wait_future_exception(mock_as_completed: MagicMock) -> None:
    """Test wait when a future raises an exception."""
    rpc_thread = RPCThreadImport(1, MagicMock(), [])
    mock_future = MagicMock()
    mock_future.result.side_effect = Exception("Future error")
    mock_as_completed.return_value = [mock_future]
    rpc_thread.futures = [mock_future]
    rpc_thread.executor = MagicMock()
    rpc_thread.wait()
    rpc_thread.executor.shutdown.assert_called_once_with(wait=True)


def test_filter_ignored_columns_empty_ignore() -> None:
    """Test _filter_ignored_columns when ignore list is empty."""
    header = ["id", "name"]
    data = [["1", "Test"]]
    new_header, new_data = _filter_ignored_columns([], header, data)
    assert new_header == header
    assert new_data == data


def test_create_batches_value_error() -> None:
    """Test _create_batches when split_by_col is not found."""
    header = ["id", "name"]
    data = [["1", "Test"]]
    batches = list(_create_batches(data, "non_existent_col", header, 1, False))
    assert batches == []


@patch("builtins.open", new_callable=mock_open)
def test_setup_fail_file_os_error(mock_open: MagicMock) -> None:
    """Test _setup_fail_file when OSError occurs."""
    mock_open.side_effect = OSError("Permission denied")
    writer, handle = _setup_fail_file("fail.csv", ["id"], False, ";", "utf-8")
    assert writer is None
    assert handle is None


@patch("odoo_data_flow.import_threaded._read_data_file", return_value=([], []))
def test_import_data_empty_file(mock_read_data_file: MagicMock) -> None:
    """Test import_data when the data file is empty."""
    result = import_data("dummy.conf", "model", file_csv="dummy.csv")
    assert result is False


def test_handle_odoo_messages_with_error_reason_generic() -> None:
    """Test _handle_odoo_messages.

    Test _handle_odoo_messages with generic messages and add_error_reason=True.
    """
    rpc_thread = RPCThreadImport(1, MagicMock(), [], add_error_reason=True)
    messages = [{"message": "Generic Error"}]
    original_lines = [["1", "Test"], ["2", "Test2"]]
    failed_lines = rpc_thread._handle_odoo_messages(messages, original_lines)
    assert failed_lines == [
        ["1", "Test", "Generic Error | "],
        ["2", "Test2", "Generic Error | "],
    ]


def test_handle_rpc_error_with_error_reason_appends_message() -> None:
    """Test _handle_rpc_error.

    Test _handle_rpc_error appends error message when add_error_reason is True.
    """
    rpc_thread = RPCThreadImport(1, MagicMock(), [], add_error_reason=True)
    error = Exception("Test RPC Error")
    lines = [["1", "data1"], ["2", "data2"]]
    failed_lines = rpc_thread._handle_rpc_error(error, lines)
    assert failed_lines == [
        ["1", "data1", "Test RPC Error"],
        ["2", "data2", "Test RPC Error"],
    ]


def test_handle_record_mismatch_with_error_reason_appends_message() -> None:
    """Test _handle_record_mismatch.

    Test _handle_record_mismatch appends error message when add_error_reason is True.
    """
    rpc_thread = RPCThreadImport(1, MagicMock(), [], add_error_reason=True)
    response = {"ids": [1]}
    lines = [["1", "data1"], ["2", "data2"]]
    failed_lines = rpc_thread._handle_record_mismatch(response, lines)
    assert failed_lines[0][2].startswith("Record count mismatch")
    assert failed_lines[1][2].startswith("Record count mismatch")


@patch("odoo_data_flow.import_threaded.RPCThreadImport.spawn_thread")
def test_launch_batch_with_failed_lines_and_writer(
    mock_spawn_thread: MagicMock,
) -> None:
    """Test launch_batch writes failed lines when writer is present."""
    mock_writer = MagicMock()
    rpc_thread = RPCThreadImport(1, MagicMock(), [], writer=mock_writer)

    # Simulate a failed batch by making model.load return messages
    with patch.object(rpc_thread, "model") as mock_model:
        mock_model.load.return_value = {"messages": [{"message": "Error", "record": 0}]}
        rpc_thread.launch_batch([["1", "Test"]], 1)

    mock_writer.writerows.assert_called_once()


def test_create_batches_value_error_split_by_col_not_found() -> None:
    """Test _create_batches.

    Test _create_batches logs error and returns empty when split_by_col is not found.
    """
    header = ["col1", "col2"]
    data = [["a", "1"], ["b", "2"]]
    with patch("odoo_data_flow.import_threaded.log") as mock_log:
        batches = list(_create_batches(data, "non_existent_col", header, 1, False))
        assert batches == []
        mock_log.error.assert_called_once_with(
            # Update this string to match the 'Actual' message exactly
            "Grouping column ''non_existent_col' is not in list' "
            "not found in header. Cannot use --groupby."
        )


@patch("odoo_data_flow.import_threaded._setup_fail_file", return_value=(None, None))
@patch("odoo_data_flow.lib.conf_lib.get_connection_from_config")
def test_import_data_setup_fail_file_fails(
    mock_get_connection: MagicMock, mock_setup_fail_file: MagicMock
) -> None:
    """Test import_data returns False if _setup_fail_file fails."""
    mock_get_connection.return_value.get_model.return_value = MagicMock()
    result = import_data(
        "dummy.conf", "model", header=["id"], data=[["1"]], fail_file="fail.csv"
    )
    assert result is False


@patch("odoo_data_flow.import_threaded.log.error")
def test_read_data_file_value_error(mock_log_error: MagicMock) -> None:
    """Test _read_data_file handles ValueError."""
    with patch("builtins.open", mock_open(read_data="id,name\n1,test")) as mock_file:
        mock_file.side_effect = ValueError("Test error")
        with pytest.raises(ValueError):
            _read_data_file("dummy.csv", ",", "utf-8", 0)
        mock_log_error.assert_called_once_with(
            f"Failed to read file dummy.csv: {ValueError('Test error')}"
        )


def test_handle_rpc_error_no_error_reason() -> None:
    """Tests that when add_error_reason is False, the reason is not appended."""
    header = ["id", "name"]
    lines = [["1", "A"], ["2", "B"]]
    mock_writer = MagicMock()
    rpc_thread = RPCThreadImport(1, None, header, mock_writer, add_error_reason=False)
    error = Exception("Connection Timed Out")
    failed_lines = rpc_thread._handle_rpc_error(error, lines)
    assert len(failed_lines) == 2
    assert len(failed_lines[0]) == 2  # No error reason appended


@patch("odoo_data_flow.import_threaded.log.error")
@patch("odoo_data_flow.import_threaded.RPCThreadImport.spawn_thread")
def test_launch_batch_fun_generic_exception(
    mock_spawn_thread: MagicMock, mock_log_error: MagicMock
) -> None:
    """Tests that a generic exception in launch_batch_fun is handled."""
    mock_model = MagicMock()
    mock_model.load.side_effect = Exception("Generic Error")
    rpc_thread = RPCThreadImport(1, mock_model, ["id"], None)

    def spawn_side_effect(
        func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        # Call the function immediately
        func(*args, **kwargs)

    mock_spawn_thread.side_effect = spawn_side_effect

    rpc_thread.launch_batch([["1"]], 1, False)

    mock_log_error.assert_called_once()
    assert (
        "RPC call for batch 1 failed: Generic Error" in mock_log_error.call_args[0][0]
    )


@patch("odoo_data_flow.import_threaded.log.error")
@patch("odoo_data_flow.import_threaded.RPCThreadImport.spawn_thread")
def test_launch_batch_fun_record_mismatch(
    mock_spawn_thread: MagicMock, mock_log_error: MagicMock
) -> None:
    """Tests that a record mismatch in launch_batch_fun is handled."""
    mock_model = MagicMock()
    mock_model.load.return_value = {"ids": [1]}  # Mismatch: 2 lines sent, 1 id back
    rpc_thread = RPCThreadImport(1, mock_model, ["id"], None, add_error_reason=True)

    def spawn_side_effect(
        func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        # Call the function immediately
        func(*args, **kwargs)

    mock_spawn_thread.side_effect = spawn_side_effect

    lines_to_send = [["1"], ["2"]]
    with patch.object(
        rpc_thread,
        "_handle_record_mismatch",
    ) as mock_handle_mismatch:
        rpc_thread.launch_batch(lines_to_send, 1, True)
        mock_handle_mismatch.assert_called_once()


@patch("odoo_data_flow.import_threaded.log.error")
@patch("odoo_data_flow.import_threaded.RPCThreadImport.spawn_thread")
def test_launch_batch_connection_error(
    mock_spawn_thread: MagicMock, mock_log_error: MagicMock
) -> None:
    """Tests that a ConnectionError in launch_batch_fun is handled."""
    mock_model = MagicMock()
    mock_model.load.side_effect = requests.exceptions.ConnectionError(
        "Connection Error"
    )
    rpc_thread = RPCThreadImport(1, mock_model, ["id"], None)

    def spawn_side_effect(
        func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        func(*args, **kwargs)

    mock_spawn_thread.side_effect = spawn_side_effect

    rpc_thread.launch_batch([["1"]], 1, False)

    assert rpc_thread.abort_flag is True
    mock_log_error.assert_called_once()
    assert (
        "Connection to Odoo failed: Connection Error" in mock_log_error.call_args[0][0]
    )


def test_read_data_file_with_skip(tmp_path: Path) -> None:
    """Tests that _read_data_file correctly skips the specified number of lines."""
    source_file = tmp_path / "test.csv"
    source_file.write_text("id,name\n1,A\n2,B\n3,C")
    header, data = _read_data_file(str(source_file), ",", "utf-8", skip=1)
    assert header == ["id", "name"]
    assert data == [["2", "B"], ["3", "C"]]


def test_create_batches_with_o2m() -> None:
    """Tests batch creation with o2m handling."""
    header = ["id", "rel_id", "value"]
    data = [
        ["rec1", "relA", "v1"],
        ["", "relA", "v2"],  # o2m line for relA
        ["rec2", "relB", "v3"],
        ["rec3", "relB", "v4"],
    ]
    batches = list(_create_batches(data, "rel_id", header, 2, o2m=True))
    assert len(batches) == 2
    assert batches[0][0] == "0-relA"
    assert len(batches[0][1]) == 2  # rec1 and its o2m line
    assert batches[1][0] == "1-relB"
    assert len(batches[1][1]) == 2  # rec2 and rec3 for relB


def test_handle_odoo_messages_record_index_out_of_bounds() -> None:
    """Tests that messages with out-of-bounds record indices are handled."""
    rpc_thread = RPCThreadImport(1, MagicMock(), [], add_error_reason=True)
    messages = [{"message": "Error", "record": 99}]  # Index 99 is out of bounds
    original_lines = [["1", "Test"]]
    failed_lines = rpc_thread._handle_odoo_messages(messages, original_lines)
    # The message is generic now, so it applies to all lines
    assert len(failed_lines) == 1
    assert failed_lines[0][-1] == "Error | "
