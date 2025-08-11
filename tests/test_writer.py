"""Test the high-level writer orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

import httpx
from rich.panel import Panel
from rich.progress import Progress, TaskID

from odoo_data_flow import writer
from odoo_data_flow.write_threaded import RPCThreadWrite
from odoo_data_flow.writer import _read_data_file, run_write


class TestRunWrite:
    """Tests for the main run_write function in writer.py."""

    @patch("odoo_data_flow.writer.write_threaded.write_data")
    def test_run_write_success_path(
        self, mock_write_data: MagicMock, tmp_path: Path
    ) -> None:
        """Tests the standard, successful execution path of run_write."""
        source_file = tmp_path / "updates.csv"
        source_file.write_text("id,name\n101,New Name")

        run_write(
            config="dummy.conf",
            filename=str(source_file),
            model="res.partner",
            fail=False,
            separator=",",
        )

        mock_write_data.assert_called_once()
        call_kwargs = mock_write_data.call_args.kwargs
        assert call_kwargs["model"] == "res.partner"
        assert call_kwargs["is_fail_run"] is False

    @patch("odoo_data_flow.writer.write_threaded.write_data")
    def test_run_write_fail_mode(
        self, mock_write_data: MagicMock, tmp_path: Path
    ) -> None:
        """Tests that --fail mode correctly reads the _write_fail.csv file."""
        source_file = tmp_path / "original.csv"
        source_file.touch()

        fail_file = tmp_path / "res_partner_write_fail.csv"
        fail_file.write_text("id,name\n102,Retry Name")

        run_write(
            config="dummy.conf",
            filename=str(source_file),
            model="res.partner",
            fail=True,
            separator=",",
        )

        mock_write_data.assert_called_once()
        call_kwargs = mock_write_data.call_args.kwargs
        assert call_kwargs["is_fail_run"] is True
        assert call_kwargs["header"] == ["id", "name"]
        assert call_kwargs["data"] == [["102", "Retry Name"]]

    patch("odoo_data_flow.writer.Console")

    @patch("odoo_data_flow.writer.Console")
    @patch("odoo_data_flow.writer.write_threaded.write_data")
    def test_run_write_fail_mode_no_records_to_retry(
        self,
        mock_write_data: MagicMock,
        mock_console_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test run fail mode no records to retry.

        Tests that a panel is displayed and the process exits if the fail
        file is empty or missing.
        """
        # 1. Setup
        source_file = tmp_path / "res_partner.csv"
        source_file.write_text("id,name\n1,test")
        (tmp_path / "res_partner_write_fail.csv").write_text("id,_ERROR_REASON\n")

        # Get a reference to the mock instance that will be created
        mock_console_instance = mock_console_class.return_value

        # 2. Action
        run_write(
            config="dummy.conf",
            filename=str(source_file),
            model="res.partner",
            fail=True,
            separator=",",
        )

        # 3. Assertions
        mock_write_data.assert_not_called()

        # The print method on our mock console instance should have been called
        mock_console_instance.print.assert_called_once()

        # Check the content of the printed panel
        panel = mock_console_instance.print.call_args[0][0]
        assert "No Recovery Needed" in str(panel.title)
        assert "Nothing to retry" in str(panel.renderable)

    def test_read_data_file_no_id_column(self, tmp_path: Path) -> None:
        """Tests that _read_data_file raises ValueError if 'id' column is missing."""
        source_file = tmp_path / "no_id.csv"
        source_file.write_text("name,value\nTest,100")

        with patch("odoo_data_flow.writer.log.error") as mock_log:
            header, data = _read_data_file(str(source_file), ",", "utf-8")
            assert header == []
            assert data == []
            mock_log.assert_called_once()
            assert "must contain an 'id' column" in mock_log.call_args[0][0]

    @patch("odoo_data_flow.writer._read_data_file")
    @patch("odoo_data_flow.writer.write_threaded.write_data")
    @patch("odoo_data_flow.writer.log")
    def test_run_write_no_data_rows(
        self,
        mock_log: MagicMock,
        mock_write_data: MagicMock,
        mock_read_data: MagicMock,
    ) -> None:
        """Tests that run_write exits gracefully if the file has no data rows."""
        # ARRANGE: Simulate _read_data_file returning a header but an empty data list.
        mock_read_data.return_value = (["id", "name"], [])

        # ACT
        writer.run_write(
            config="conf.ini",
            filename="data.csv",
            model="res.partner",
            fail=False,
        )

        # ASSERT
        mock_log.warning.assert_called_with(
            "No data rows found in the source file. Nothing to write."
        )
        mock_write_data.assert_not_called()

    @patch("odoo_data_flow.writer._read_data_file")
    @patch("odoo_data_flow.writer.write_threaded.write_data")
    @patch("rich.console.Console.print")
    def test_run_write_handles_failure(
        self,
        mock_print: MagicMock,
        mock_write_data: MagicMock,
        mock_read_data: MagicMock,
    ) -> None:
        """Tests that run_write prints a failure panel when the write process fails."""
        # ARRANGE
        mock_read_data.return_value = (["id", "name"], [["1", "Test"]])
        mock_write_data.return_value = False

        # ACT
        writer.run_write(
            config="conf.ini",
            filename="data.csv",
            model="res.partner",
            fail=False,
        )

        # ASSERT
        mock_write_data.assert_called_once()

        assert mock_print.called

        final_output = mock_print.call_args.args[0]

        assert isinstance(final_output, Panel)

        assert final_output.title is not None
        assert "Write Failed" in final_output.title

    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_read_data_file_empty(self, mock_file: MagicMock) -> None:
        """Tests _read_data_file with a completely empty file."""
        # ARRANGE: The mock file is empty, which will cause a StopIteration.

        # ACT
        header, data = writer._read_data_file("empty.csv", ";", "utf-8")

        # ASSERT
        assert header == []
        assert data == []


class TestRPCThreadWrite:
    """Tests for the RPCThreadWrite class in write_threaded.py."""

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

    def test_execute_batch_json_decode_error(self) -> None:
        """Tests graceful handling of a JSONDecodeError."""
        mock_model = MagicMock()
        mock_model.write.side_effect = httpx.DecodingError("Expecting value", request=None)
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

    @patch("odoo_data_flow.lib.internal.rpc_thread.RpcThread.wait")
    def test_wait_fallback_without_progress(self, mock_super_wait: MagicMock) -> None:
        """Tests that wait() calls super().wait() if no progress bar is given."""
        rpc_thread = RPCThreadWrite(1, MagicMock(), [])
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

    def test_read_data_file_not_found(self) -> None:
        """Tests that _read_data_file handles a FileNotFoundError."""
        with patch("odoo_data_flow.writer.log.error") as mock_log:
            header, data = _read_data_file("non_existent_file.csv", ",", "utf-8")
            assert header == []
            assert data == []
            mock_log.assert_called_once()
            assert "Source file not found" in mock_log.call_args[0][0]

    @patch("odoo_data_flow.writer.write_threaded.write_data")
    def test_run_write_empty_file(
        self, mock_write_data: MagicMock, tmp_path: Path
    ) -> None:
        """Tests that the process exits gracefully if the source file is empty."""
        source_file = tmp_path / "updates.csv"
        source_file.write_text("id,name\n")  # Header only

        run_write(
            config="dummy.conf",
            filename=str(source_file),
            model="res.partner",
            fail=False,
            separator=",",
        )
        mock_write_data.assert_not_called()

    @patch("odoo_data_flow.writer.Console")
    @patch("odoo_data_flow.writer.write_threaded.write_data", return_value=False)
    def test_run_write_prints_failure_panel(
        self,
        mock_write_data: MagicMock,
        mock_console_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Tests that the failure panel is shown when write_data returns False."""
        source_file = tmp_path / "updates.csv"
        source_file.write_text("id,name\n101,New Name")

        run_write(
            config="dummy.conf",
            filename=str(source_file),
            model="res.partner",
            fail=False,
            separator=",",
        )

        mock_write_data.assert_called_once()
