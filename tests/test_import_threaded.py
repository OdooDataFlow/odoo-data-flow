"""Test the low-level, multi-threaded import logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from odoo_data_flow.import_threaded import (
    RPCThreadImport,
    _create_batches,
    _read_data_file,
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
        assert failed_lines[0][-1] == "Generic Error | "

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
        assert len(failed_lines[0]) == 2  # No extra column added

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

    @patch("odoo_data_flow.import_threaded.log.error")
    def test_read_data_file_no_id_column(
        self, mock_log_error: MagicMock, tmp_path: Path
    ) -> None:
        """Tests that _read_data_file logs an error if 'id' column is missing."""
        source_file = tmp_path / "no_id.csv"
        source_file.write_text("name,value\nTest,100")
        header, data = _read_data_file(str(source_file), ",", "utf-8", 0)
        assert header == []
        assert data == []
        mock_log_error.assert_called_once()
        assert "Failed to read file" in mock_log_error.call_args[0][0]

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
        import_data(
            config_file="bad.conf",
            model="dummy.model",
            header=["id"],
            data=[["1"]],
        )
        mock_get_conn.assert_called_once()

    @patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_config")
    @patch("odoo_data_flow.import_threaded.open")
    def test_import_data_fail_file_oserror(
        self, mock_open: MagicMock, mock_get_conn: MagicMock
    ) -> None:
        """Tests that the function handles an OSError when opening the fail file."""
        mock_get_conn.return_value = MagicMock()
        mock_open.side_effect = OSError("Permission denied")
        import_data(
            config_file="dummy.conf",
            model="dummy.model",
            header=["id"],
            data=[["1"]],
            fail_file="protected/fail.csv",
        )
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
