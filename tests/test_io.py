"""Test the IO Handling functionalities."""

# tests/test_io.py

import shlex
from pathlib import Path
from unittest.mock import MagicMock, patch

from odoo_data_flow.lib.internal.io import write_csv, write_file

# --- Tests for write_csv ---


@patch("odoo_data_flow.lib.internal.io.open")
@patch("odoo_data_flow.lib.internal.io.log.error")
def test_write_csv_oserror(mock_log_error: MagicMock, mock_open: MagicMock) -> None:
    """Tests that write_csv logs an error if an OSError occurs."""
    # 1. Setup: Make the open call raise an OSError
    mock_open.side_effect = OSError("Permission denied")

    # 2. Action
    write_csv("protected/file.csv", ["h1"], [["d1"]])

    # 3. Assertions
    mock_log_error.assert_called_once()
    assert "Failed to write to file" in mock_log_error.call_args[0][0]


# --- Tests for write_file ---


def test_write_file_writes_csv_data(tmp_path: Path) -> None:
    """Tests that write_file correctly calls write_csv to create the data file."""
    data_file = tmp_path / "data.csv"

    with patch("odoo_data_flow.lib.internal.io.write_csv") as mock_write_csv:
        write_file(
            filename=str(data_file),
            header=["id", "name"],
            data=[["1", "test"]],
            launchfile="",  # Correctly pass an empty string instead of None
        )
        mock_write_csv.assert_called_once_with(
            str(data_file), ["id", "name"], [["1", "test"]], encoding="utf-8"
        )


@patch("odoo_data_flow.lib.internal.io.write_csv")  # Mock the CSV writing part
@patch("odoo_data_flow.lib.internal.io.open")
def test_write_file_no_launchfile(
    mock_open: MagicMock, mock_write_csv: MagicMock, tmp_path: Path
) -> None:
    """Tests that write_file exits early if no launchfile is specified."""
    data_file = tmp_path / "data.csv"

    write_file(
        filename=str(data_file),
        header=["id"],
        data=[["1"]],
        launchfile="",  # Empty string means no script
    )

    # Assert that write_csv was called, but open was not (for the launchfile)
    mock_write_csv.assert_called_once()
    mock_open.assert_not_called()


def test_write_file_full_script_generation(tmp_path: Path) -> None:
    """Tests that write_file generates a complete shell script with all options."""
    # 1. Setup
    script_file = tmp_path / "load.sh"
    data_file = tmp_path / "my_model.csv"

    # 2. Action
    write_file(
        filename=str(data_file),
        header=["id", "name"],
        data=[["1", "test"]],
        launchfile=str(script_file),
        model="my.model",
        fail=True,
        init=True,
        worker=4,
        batch_size=50,
        groupby="parent_id/id",
        ignore="field_to_ignore",
        context={"active_test": False},  # Correctly pass a dict instead of a string
        conf_file="conf/custom.conf",
    )

    # 3. Assertions
    assert script_file.exists()
    content = script_file.read_text()

    # Check for the main command
    assert "odoo-data-flow import" in content
    assert f"--config {shlex.quote('conf/custom.conf')}" in content
    assert f"--file {shlex.quote(str(data_file))}" in content
    assert f"--model {shlex.quote('my.model')}" in content
    assert "--worker 4" in content
    assert "--size 50" in content
    assert f"--groupby {shlex.quote('parent_id/id')}" in content
    assert f"--ignore {shlex.quote('field_to_ignore')}" in content
    assert f"--context {shlex.quote(str({'active_test': False}))}" in content

    # Check for the second command with the --fail flag
    assert "--fail" in content
    # Count occurrences to ensure both commands are present
    assert content.count("odoo-data-flow import") == 2


def test_write_file_auto_model_name(tmp_path: Path) -> None:
    """Tests that the model name is correctly inferred when model='auto'."""
    script_file = tmp_path / "load_auto.sh"
    data_file = tmp_path / "res.partner.csv"

    write_file(
        filename=str(data_file),
        header=["id"],
        data=[["1"]],
        launchfile=str(script_file),
        model="auto",
        init=True,
    )

    content = script_file.read_text()
    # The model name should be inferred from 'res.partner.csv' -> 'res.partner'
    assert f"--model {shlex.quote('res.partner')}" in content


@patch("odoo_data_flow.lib.internal.io.write_csv")  # Mock the CSV part
@patch("odoo_data_flow.lib.internal.io.open")
@patch("odoo_data_flow.lib.internal.io.log.error")
def test_write_file_oserror(
    mock_log_error: MagicMock, mock_open: MagicMock, mock_write_csv: MagicMock
) -> None:
    """Test write fle os error.

    Tests that write_file logs an error if an OSError occurs during script writing.
    """
    # 1. Setup: This time, the 'open' for the launchfile will fail
    mock_open.side_effect = OSError("Permission denied on script file")

    # 2. Action
    write_file(
        filename="data.csv",
        header=["id"],
        data=[["1"]],
        launchfile="protected/load.sh",
        init=True,
    )

    # 3. Assertions
    mock_write_csv.assert_called_once()  # Ensure the CSV part was attempted
    mock_log_error.assert_called_once()
    assert "Failed to write to launch file" in mock_log_error.call_args[0][0]
