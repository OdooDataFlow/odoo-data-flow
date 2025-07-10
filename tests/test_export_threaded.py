"""Test the Export Handling mechanism."""

# tests/test_export_threaded.py

import csv
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, patch

from odoo_data_flow.export_threaded import (
    RPCThreadExport,
    _fetch_export_data,
    export_data_for_migration,
    export_data_to_file,
)


def test_export_data_to_file_writes_file(tmp_path: Path) -> None:
    """Tests the main export_data function when writing to a file.

    This test verifies that:
    1. It correctly connects and searches for records.
    2. It processes multiple batches.
    3. It writes the header and all data rows to the output CSV file.
    """
    # 1. Setup: Mock the Odoo connection and define test data
    output_file = tmp_path / "export_output.csv"
    model_name = "res.partner"
    header = ["id", "name"]

    # This mock simulates the odoo-client-lib connection
    mock_connection = MagicMock()
    mock_model_obj = MagicMock()

    # Simulate Odoo's search method returning a list of IDs
    mock_model_obj.search.return_value = [1, 2, 3, 4, 5]

    # Simulate Odoo's export_data method returning different data for each call
    mock_model_obj.export_data.side_effect = [
        {"datas": [["1", "Partner A"], ["2", "Partner B"]]},  # Batch 1
        {"datas": [["3", "Partner C"], ["4", "Partner D"]]},  # Batch 2
        {"datas": [["5", "Partner E"]]},  # Batch 3
    ]
    mock_connection.get_model.return_value = mock_model_obj

    with patch(
        "odoo_data_flow.export_threaded.conf_lib.get_connection_from_config",
        return_value=mock_connection,
    ):
        success, message = export_data_to_file(
            config_file="dummy.conf",
            model=model_name,
            domain=[("is_company", "=", True)],
            header=header,
            output=str(output_file),
            batch_size=2,  # Use a small batch size to test batching logic
            separator=",",
        )

    # 3. Assertions
    assert success is True
    assert message == "Export complete."
    assert output_file.exists(), "Output file was not created."

    with open(output_file, encoding="utf-8") as f:
        reader = csv.reader(f)
        result_header = next(reader)
        result_data = list(reader)

    assert result_header == header
    assert len(result_data) == 5
    assert result_data[0] == ["1", "Partner A"]
    assert result_data[4] == ["5", "Partner E"]

    # Verify that search and export_data were called correctly
    mock_model_obj.search.assert_called_once()
    assert mock_model_obj.export_data.call_count == 3


def test_export_data_for_migration_returns_data() -> None:
    """Tests the main export_data function when returning data in-memory."""
    # 1. Setup
    mock_connection = MagicMock()
    mock_model_obj = MagicMock()
    mock_model_obj.search.return_value = [1, 2]
    mock_model_obj.export_data.return_value = {
        "datas": [["1", "Mem Partner"], ["2", "Mem Partner 2"]]
    }
    mock_connection.get_model.return_value = mock_model_obj

    # 2. Action
    with patch(
        "odoo_data_flow.export_threaded.conf_lib.get_connection_from_config",
        return_value=mock_connection,
    ):
        header, data = export_data_for_migration(
            config_file="dummy.conf",
            model="res.partner",
            domain=[],
            header=["id", "name"],
        )

    # 3. Assertions
    assert header == ["id", "name"]
    assert data is not None
    assert len(data) == 2
    assert data[0] == ["1", "Mem Partner"]


def test_export_data_for_migration_connection_failure() -> None:
    """Tests that the export function handles a connection failure gracefully."""
    # 1. Setup: This time, the get_connection call will raise an exception
    with patch(
        "odoo_data_flow.export_threaded.conf_lib.get_connection_from_config",
        side_effect=Exception("Connection failed"),
    ) as mock_get_conn:
        # 2. Action
        header, data = export_data_for_migration(
            config_file="bad.conf",
            model="res.partner",
            domain=[],
            header=["name"],
        )

        # 3. Assertions
        mock_get_conn.assert_called_once()
        assert header == ["name"]
        assert data is None


def test_export_data_to_file_connection_failure() -> None:
    """Tests that the export function handles a connection failure gracefully."""
    # 1. Setup: This time, the get_connection call will raise an exception
    with patch(
        "odoo_data_flow.export_threaded.conf_lib.get_connection_from_config",
        side_effect=Exception("Connection failed"),
    ) as mock_get_conn:
        # 2. Action
        success, message = export_data_to_file(
            config_file="bad.conf",
            model="res.partner",
            domain=[],
            header=["name"],
            output="foo.csv",
        )

        # 3. Assertions
        mock_get_conn.assert_called_once()
        assert success is False
        assert "Failed to connect" in message


def test_rpc_thread_export_no_context() -> None:
    """Tests RPCThreadExport initialization with context=None."""
    thread = RPCThreadExport(1, MagicMock(), ["id"], context=None)
    assert thread.context == {}


@patch("odoo_data_flow.export_threaded.log")
def test_rpc_thread_export_launch_batch_exception(mock_log: MagicMock) -> None:
    """Tests exception handling in the launch_batch function."""
    mock_model = MagicMock()
    mock_model.export_data.side_effect = Exception("Odoo Error")
    thread = RPCThreadExport(1, mock_model, ["id"])
    thread.launch_batch([1], 0)
    thread.wait()
    mock_log.error.assert_called_once()
    assert "Export for batch 0 failed" in mock_log.error.call_args[0][0]


@patch("odoo_data_flow.export_threaded.conf_lib.get_connection_from_config")
@patch("builtins.open")
def test_export_data_to_file_os_error(
    mock_open: MagicMock, mock_get_connection: MagicMock
) -> None:
    """Tests that export_data_to_file handles an OSError during file write."""
    mock_get_connection.return_value = MagicMock()
    mock_open.side_effect = OSError("Disk full")

    with patch(
        "odoo_data_flow.export_threaded._fetch_export_data",
        return_value=[["data"]],
    ):
        success, message = export_data_to_file(
            config_file="dummy.conf",
            model="res.partner",
            domain=[],
            header=["name"],
            output="some/path/that/fails.csv",
        )

    assert success is False
    assert "Failed to write to output file" in message


@patch("odoo_data_flow.export_threaded.RPCThreadExport.spawn_thread")
@patch("odoo_data_flow.export_threaded.conf_lib.get_connection_from_config")
def test_fetch_export_data_with_technical_names(
    mock_get_connection: MagicMock, mock_spawn_thread: MagicMock
) -> None:
    """Tests that _fetch_export_data uses model.read with technical_names=True."""
    mock_connection = MagicMock()
    mock_model_obj = MagicMock()
    mock_get_connection.return_value = mock_connection
    mock_connection.get_model.return_value = mock_model_obj

    header = ["id", "name", "type"]
    mock_model_obj.search.return_value = [1, 2]
    mock_model_obj.read.return_value = [
        {"id": 1, "name": "Partner 1", "type": "delivery"},
        {"id": 2, "name": "Partner 2", "type": "invoice"},
    ]

    def call_directly(fun: Callable[..., Any], args: list[Any]) -> None:
        fun(*args)

    mock_spawn_thread.side_effect = call_directly

    with (
        patch("odoo_data_flow.export_threaded.Progress"),
        patch("odoo_data_flow.export_threaded.concurrent.futures"),
    ):
        result = _fetch_export_data(
            connection=mock_connection,
            model_name="res.partner",
            domain=[],
            header=header,
            context={},
            max_connection=1,
            batch_size=10,
            technical_names=True,
        )

    mock_model_obj.read.assert_called_once_with([1, 2], header)
    assert not mock_model_obj.export_data.called
    assert result == [[1, "Partner 1", "delivery"], [2, "Partner 2", "invoice"]]
