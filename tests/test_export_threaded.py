"""Test the Export Handling mechanism."""

# tests/test_export_threaded.py

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

from odoo_data_flow.export_threaded import export_data


def test_export_data_to_file(tmp_path: Path) -> None:
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

    # 2. Action: Run the export function
    with patch(
        "odoo_data_flow.export_threaded.conf_lib.get_connection_from_config",
        return_value=mock_connection,
    ):
        export_data(
            config_file="dummy.conf",
            model=model_name,
            domain=[("is_company", "=", True)],
            header=header,
            output=str(output_file),
            batch_size=2,  # Use a small batch size to test batching logic
            separator=",",
        )

    # 3. Assertions
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


def test_export_data_in_memory() -> None:
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
        header, data = export_data(
            config_file="dummy.conf",
            model="res.partner",
            domain=[],
            header=["id", "name"],
            output=None,  # This signals the function to return data
        )

    # 3. Assertions
    assert header == ["id", "name"]
    assert data is not None
    assert len(data) == 2
    assert data[0] == ["1", "Mem Partner"]


def test_export_data_connection_failure() -> None:
    """Tests that the export function handles a connection failure gracefully."""
    # 1. Setup: This time, the get_connection call will raise an exception
    with patch(
        "odoo_data_flow.export_threaded.conf_lib.get_connection_from_config",
        side_effect=Exception("Connection failed"),
    ) as mock_get_conn:
        # 2. Action
        header, data = export_data(
            config_file="bad.conf",
            model="res.partner",
            domain=[],
            header=["name"],
        )

        # 3. Assertions
        mock_get_conn.assert_called_once()
        assert header is None
        assert data is None
