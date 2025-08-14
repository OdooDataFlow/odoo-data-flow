"""Test the writer low-level threaded logic."""

import importlib
import sys
from typing import Optional
from unittest.mock import MagicMock, call, mock_open, patch

import httpx
import pytest
from rich.progress import Progress, TaskID

from odoo_data_flow import write_threaded
from odoo_data_flow.write_threaded import RPCThreadWrite, write_data


@patch("csv.field_size_limit")
def test_csv_field_size_limit_overflow(
    mock_field_size_limit: MagicMock,
) -> None:
    """Tests that the OverflowError for csv.field_size_limit is handled."""
    # Configure the mock to raise OverflowError on the first call
    mock_field_size_limit.side_effect = [OverflowError, None]

    # Reload the module to execute the top-level code again
    importlib.reload(write_threaded)

    # Check that it was called first with sys.maxsize, and then with the fallback
    mock_field_size_limit.assert_has_calls([call(sys.maxsize), call(2**30)])
    assert mock_field_size_limit.call_count == 2


class TestRPCThreadWrite:
    """Tests for the RPCThreadWrite class in write_threaded.py."""

    def test_init_with_none_context(self) -> None:
        """Tests that context defaults to an empty dict if None is passed."""
        rpc_thread = RPCThreadWrite(1, MagicMock(), [], context=None)
        assert rpc_thread.context == {}

    def test_execute_batch_grouping(self) -> None:
        """Tests that records with identical values are grouped into one RPC call."""
        mock_model = MagicMock()
        header = ["id", "active", "comment"]
        lines = [
            ["101", "False", ""],
            ["102", "False", ""],
            ["103", "True", "Needs review"],
        ]
        rpc_thread = RPCThreadWrite(1, mock_model, header)

        rpc_thread._execute_batch(lines, 1)

        assert mock_model.write.call_count == 2
        mock_model.write.assert_has_calls(
            [
                call([101, 102], {"active": "False", "comment": ""}),
                call([103], {"active": "True", "comment": "Needs review"}),
            ],
            any_order=True,
        )

    def test_execute_batch_aborted(self) -> None:
        """Tests that the batch execution aborts if the abort_flag is set."""
        mock_model = MagicMock()
        rpc_thread = RPCThreadWrite(1, mock_model, ["id"])
        rpc_thread.abort_flag = True

        result = rpc_thread._execute_batch([["1"]], 1)

        assert result["error_summary"] == "Aborted"
        assert result["processed"] == 0
        mock_model.write.assert_not_called()

    def test_execute_batch_grouping_error(self) -> None:
        """Tests handling of an exception during the grouping phase."""
        mock_model = MagicMock()
        # Header is missing the 'id' column, which will cause a ValueError
        header = ["name"]
        lines = [["Test"]]
        rpc_thread = RPCThreadWrite(1, mock_model, header)

        result = rpc_thread._execute_batch(lines, 1)

        assert result["failed"] == 1
        assert "'id' is not in list" in result["error_summary"]

    def test_execute_batch_json_decode_error(self) -> None:
        """Tests graceful handling of a JSONDecodeError."""
        mock_model = MagicMock()
        mock_model.write.side_effect = httpx.DecodingError(
            "Expecting value", request=None
        )
        header = ["id", "active"]
        lines = [["101", "False"]]
        rpc_thread = RPCThreadWrite(1, mock_model, header)

        with patch("odoo_data_flow.write_threaded.log.error") as mock_log:
            result = rpc_thread._execute_batch(lines, 1)
            assert result["failed"] == 1
            mock_log.assert_called_once()
            assert "Likely a proxy timeout" in mock_log.call_args[0][0]

    def test_execute_batch_generic_exception(self) -> None:
        """Tests handling of a generic exception during a write call."""
        mock_model = MagicMock()
        mock_model.write.side_effect = Exception("Odoo Error")
        header = ["id", "active"]
        lines = [["101", "False"]]
        mock_writer = MagicMock()
        rpc_thread = RPCThreadWrite(1, mock_model, header, writer=mock_writer)

        result = rpc_thread._execute_batch(lines, 1)

        assert result["failed"] == 1
        assert result["error_summary"] == "Odoo Error"
        mock_writer.writerow.assert_called_once_with([101, "Odoo Error"])

    def test_launch_batch_aborted(self) -> None:
        """Tests that launch_batch does nothing if the abort_flag is set."""
        rpc_thread = RPCThreadWrite(1, MagicMock(), [])
        rpc_thread.abort_flag = True
        with patch.object(rpc_thread, "spawn_thread") as mock_spawn:
            rpc_thread.launch_batch([["data"]], 1)
            mock_spawn.assert_not_called()

    @pytest.mark.parametrize(
        "progress, task_id", [(None, TaskID(1)), (Progress(), None)]
    )
    @patch("odoo_data_flow.lib.internal.rpc_thread.RpcThread.wait")
    def test_wait_fallback_without_progress(
        self,
        mock_super_wait: MagicMock,
        progress: Optional[Progress],
        task_id: Optional[TaskID],
    ) -> None:
        """Tests that wait() calls super().wait() if progress or task_id is missing."""
        rpc_thread = RPCThreadWrite(
            1, MagicMock(), [], progress=progress, task_id=task_id
        )
        rpc_thread.wait()
        mock_super_wait.assert_called_once()

    def test_wait_updates_progress_bar(self) -> None:
        """Tests that wait() updates the progress bar on task completion."""
        mock_progress = MagicMock(spec=Progress)
        mock_task_id = MagicMock(spec=TaskID)
        rpc_thread = RPCThreadWrite(
            1, MagicMock(), [], progress=mock_progress, task_id=mock_task_id
        )

        future = MagicMock()
        future.result.return_value = {
            "processed": 5,
            "error_summary": "An Error",
        }
        rpc_thread.futures = [future]

        with patch("concurrent.futures.as_completed", return_value=[future]):
            rpc_thread.wait()

        mock_progress.update.assert_called_once()
        update_kwargs = mock_progress.update.call_args.kwargs
        assert update_kwargs["advance"] == 5
        assert "Last Error: An Error" in update_kwargs["last_error"]

    def test_wait_truncates_long_error_message(self) -> None:
        """Tests that long error messages are truncated in the progress bar update."""
        mock_progress = MagicMock(spec=Progress)
        mock_task_id = MagicMock(spec=TaskID)
        rpc_thread = RPCThreadWrite(
            1, MagicMock(), [], progress=mock_progress, task_id=mock_task_id
        )

        long_error = "a" * 100
        future = MagicMock()
        future.result.return_value = {
            "processed": 1,
            "error_summary": long_error,
        }
        rpc_thread.futures = [future]

        with patch("concurrent.futures.as_completed", return_value=[future]):
            rpc_thread.wait()

        mock_progress.update.assert_called_once()
        update_kwargs = mock_progress.update.call_args.kwargs
        truncated_error = update_kwargs["last_error"]
        assert len(truncated_error) < 85
        assert truncated_error.endswith("...")

    def test_wait_handles_future_exception(self) -> None:
        """Tests that wait() logs an error if a future raises an exception."""
        mock_progress = MagicMock(spec=Progress)
        mock_task_id = MagicMock(spec=TaskID)
        rpc_thread = RPCThreadWrite(
            1, MagicMock(), [], progress=mock_progress, task_id=mock_task_id
        )

        future = MagicMock()
        future.result.side_effect = ValueError("Worker failed")
        rpc_thread.futures = [future]

        with (
            patch("odoo_data_flow.write_threaded.log.error") as mock_log,
            patch("concurrent.futures.as_completed", return_value=[future]),
        ):
            rpc_thread.wait()
            mock_log.assert_called_once()
            assert "A worker thread failed unexpectedly" in mock_log.call_args[0][0]

    def test_wait_aborts_early(self) -> None:
        """Tests that wait() aborts immediately if the abort flag is already set."""
        mock_progress = MagicMock(spec=Progress)
        mock_task_id = MagicMock(spec=TaskID)
        rpc_thread = RPCThreadWrite(
            1, MagicMock(), [], progress=mock_progress, task_id=mock_task_id
        )

        rpc_thread.executor = MagicMock()
        rpc_thread.futures = [MagicMock()]
        rpc_thread.abort_flag = True

        with patch("concurrent.futures.as_completed", return_value=rpc_thread.futures):
            rpc_thread.wait()

        rpc_thread.executor.shutdown.assert_called_once_with(
            wait=True, cancel_futures=True
        )

    def test_execute_batch_generic_exception_no_writer(self) -> None:
        """Tests handling a generic exception when no fail file writer is provided."""
        mock_model = MagicMock()
        mock_model.write.side_effect = Exception("Odoo Error")
        header = ["id", "active"]
        lines = [["101", "False"]]
        # No writer is passed, so the if-check for self.writer will be false
        rpc_thread = RPCThreadWrite(1, mock_model, header, writer=None)

        result = rpc_thread._execute_batch(lines, 1)

        # The test verifies the branch is handled correctly without crashing
        assert result["failed"] == 1
        assert result["error_summary"] == "Odoo Error"


class TestWriteData:
    """Tests for the main write_data orchestrator function."""

    @patch("odoo_data_flow.write_threaded.RPCThreadWrite")
    @patch("odoo_data_flow.write_threaded.Progress")
    @patch("builtins.open", new_callable=mock_open)
    @patch("odoo_data_flow.write_threaded.conf_lib")
    def test_write_data_happy_path(
        self,
        mock_conf: MagicMock,
        mock_open_file: MagicMock,
        mock_progress: MagicMock,
        mock_rpc_thread: MagicMock,
    ) -> None:
        """Tests the successful execution of write_data."""
        # Arrange
        mock_connection = MagicMock()
        mock_connection.get_model.return_value = MagicMock()
        mock_conf.get_connection_from_config.return_value = mock_connection
        mock_rpc_instance = mock_rpc_thread.return_value
        future = MagicMock(done=lambda: True, cancelled=lambda: False)
        future.result.return_value = {"failed": 0}
        mock_rpc_instance.futures = [future]

        # Act
        result = write_data("conf.ini", "res.partner", [], [["1"]], "fails.csv")

        # Assert
        assert result is True
        mock_conf.get_connection_from_config.assert_called_once_with("conf.ini")
        mock_open_file.assert_called_once_with(
            "fails.csv", "w", newline="", encoding="utf-8"
        )
        mock_rpc_instance.launch_batch.assert_called_once()
        mock_rpc_instance.wait.assert_called_once()

    @patch("odoo_data_flow.write_threaded.conf_lib")
    def test_write_data_connection_fails(self, mock_conf: MagicMock) -> None:
        """Tests that write_data returns False if Odoo connection fails."""
        mock_conf.get_connection_from_config.side_effect = Exception("Conn Error")
        result = write_data("conf.ini", "res.partner", [], [], "")
        assert result is False

    @patch("odoo_data_flow.write_threaded.conf_lib")
    @patch("builtins.open")
    def test_write_data_fail_file_fails(
        self, mock_open_file: MagicMock, mock_conf: MagicMock
    ) -> None:
        """Tests that write_data returns False if the fail file cannot be opened."""
        mock_conf.get_connection_from_config.return_value = MagicMock()
        mock_open_file.side_effect = OSError("Disk full")
        result = write_data("conf.ini", "res.partner", [], [], "fails.csv")
        assert result is False

    @patch("odoo_data_flow.write_threaded.RPCThreadWrite")
    @patch("odoo_data_flow.write_threaded.Progress")
    @patch("builtins.open", new_callable=mock_open)
    @patch("odoo_data_flow.write_threaded.conf_lib")
    def test_write_data_with_failures(
        self,
        mock_conf: MagicMock,
        mock_open_file: MagicMock,
        mock_progress: MagicMock,
        mock_rpc_thread: MagicMock,
    ) -> None:
        """Tests that write_data returns False if there are failed recordsa."""
        mock_conf.get_connection_from_config.return_value = MagicMock()
        mock_rpc_instance = mock_rpc_thread.return_value
        future = MagicMock(done=lambda: True, cancelled=lambda: False)
        future.result.return_value = {"failed": 1}
        mock_rpc_instance.futures = [future]

        result = write_data("conf.ini", "res.partner", [], [["1"]], "fails.csv")

        assert result is False

    @patch("odoo_data_flow.write_threaded.RPCThreadWrite")
    @patch("odoo_data_flow.write_threaded.Progress")
    @patch("odoo_data_flow.write_threaded.conf_lib")
    def test_write_data_no_fail_file(
        self,
        mock_conf: MagicMock,
        mock_progress: MagicMock,
        mock_rpc_thread: MagicMock,
    ) -> None:
        """Tests write_data when no fail_file is provided."""
        mock_conf.get_connection_from_config.return_value = MagicMock()
        mock_rpc_instance = mock_rpc_thread.return_value
        mock_rpc_instance.futures = []

        with patch("builtins.open") as mock_open_file:
            result = write_data("conf.ini", "res.partner", [], [["1"]], fail_file="")
            assert result is True
            mock_open_file.assert_not_called()
            # Check that the writer passed to RPCThreadWrite is None
            assert mock_rpc_thread.call_args.args[3] is None

    @patch("odoo_data_flow.write_threaded.RPCThreadWrite")
    @patch("odoo_data_flow.write_threaded.Progress")
    @patch("odoo_data_flow.write_threaded.conf_lib")
    def test_write_data_no_data(
        self,
        mock_conf: MagicMock,
        mock_progress: MagicMock,
        mock_rpc_thread: MagicMock,
    ) -> None:
        """Tests that write_data handles an empty data list gracefully."""
        mock_conf.get_connection_from_config.return_value = MagicMock()
        mock_rpc_instance = mock_rpc_thread.return_value
        mock_rpc_instance.futures = []

        result = write_data("conf.ini", "res.partner", [], data=[], fail_file="")

        assert result is True
        mock_rpc_instance.launch_batch.assert_not_called()

    @patch("odoo_data_flow.write_threaded.RPCThreadWrite")
    @patch("odoo_data_flow.write_threaded.Progress")
    @patch("odoo_data_flow.write_threaded.conf_lib")
    def test_write_data_ignores_undone_futures(
        self,
        mock_conf: MagicMock,
        mock_progress: MagicMock,
        mock_rpc_thread: MagicMock,
    ) -> None:
        """Tests that the final count ignores futures that are not 'done'."""
        mock_conf.get_connection_from_config.return_value = MagicMock()
        mock_rpc_instance = mock_rpc_thread.return_value

        # Simulate one future that has a failure but is not "done" yet.
        # This will test the `if f.done()` part of the filter.
        future = MagicMock(done=lambda: False)
        future.result.return_value = {"failed": 1}
        mock_rpc_instance.futures = [future]

        # The result should be True because the undone future is skipped.
        result = write_data("conf.ini", "res.partner", [], [["1"]], fail_file="")
        assert result is True

    @patch("odoo_data_flow.write_threaded.RPCThreadWrite")
    @patch("odoo_data_flow.write_threaded.Progress")
    @patch("odoo_data_flow.write_threaded.conf_lib")
    def test_write_data_ignores_cancelled_futures(
        self,
        mock_conf: MagicMock,
        mock_progress: MagicMock,
        mock_rpc_thread: MagicMock,
    ) -> None:
        """Tests that the final count ignores futures that were cancelled."""
        mock_conf.get_connection_from_config.return_value = MagicMock()
        mock_rpc_instance = mock_rpc_thread.return_value

        # Simulate a future that is done but was cancelled.
        # This will test the `and not f.cancelled()` part of the filter.
        future = MagicMock(done=lambda: True, cancelled=lambda: True)
        future.result.return_value = {"failed": 1}
        mock_rpc_instance.futures = [future]

        # The result should be True because the cancelled future is skipped.
        result = write_data("conf.ini", "res.partner", [], [["1"]], fail_file="")
        assert result is True
