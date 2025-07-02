"""Test the IO Handling functionalities."""

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
            launchfile="",
        )
        mock_write_csv.assert_called_once_with(
            str(data_file), ["id", "name"], [["1", "test"]], encoding="utf-8"
        )


@patch("odoo_data_flow.lib.internal.io.write_csv")
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
        launchfile="",
    )
    mock_write_csv.assert_called_once()
    mock_open.assert_not_called()


def test_write_file_import_command(tmp_path: Path) -> None:
    """Tests that write_file generates a complete import shell script."""
    script_file = tmp_path / "load.sh"
    data_file = tmp_path / "my_model.csv"
    write_file(
        filename=str(data_file),
        header=["id", "name"],
        data=[["1", "test"]],
        launchfile=str(script_file),
        command="import",
        model="my.model",
        fail=True,
        init=True,
        worker=4,
        batch_size=50,
        groupby="parent_id/id",
        ignore="field_to_ignore",
        context={"active_test": False},
        conf_file="conf/custom.conf",
    )
    assert script_file.exists()
    content = script_file.read_text()
    assert "odoo-data-flow import" in content
    assert f"--config {shlex.quote('conf/custom.conf')}" in content
    assert f"--file {shlex.quote(str(data_file))}" in content
    assert f"--model {shlex.quote('my.model')}" in content
    assert "--worker 4" in content
    assert "--size 50" in content
    assert f"--groupby {shlex.quote('parent_id/id')}" in content
    assert f"--ignore {shlex.quote('field_to_ignore')}" in content
    assert f"--context {shlex.quote(str({'active_test': False}))}" in content
    assert "--fail" in content
    assert content.count("odoo-data-flow import") == 2


def test_write_file_export_command(tmp_path: Path) -> None:
    """Tests that write_file generates a correct export shell script."""
    script_file = tmp_path / "export.sh"
    data_file = tmp_path / "partner_export.csv"
    domain_str = "[('is_company', '=', True)]"
    write_file(
        filename=str(data_file),
        launchfile=str(script_file),
        command="export",
        model="res.partner",
        fields="id,name",
        domain=domain_str,
        init=True,
    )
    assert script_file.exists()
    content = script_file.read_text()
    assert "odoo-data-flow export" in content
    assert f"--model {shlex.quote('res.partner')}" in content
    assert f"--fields {shlex.quote('id,name')}" in content
    expected_domain_str = f"--domain {shlex.quote(domain_str)}"
    assert expected_domain_str in content
    assert "--fail" not in content


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
        # Provide default worker and batch_size to avoid TypeError
        worker=1,
        batch_size=10,
    )
    content = script_file.read_text()
    # The model name should be inferred from 'res_partner.csv' -> 'res.partner'
    assert f"--model {shlex.quote('res.partner')}" in content


@patch("odoo_data_flow.lib.internal.io.write_csv")
@patch("odoo_data_flow.lib.internal.io.open")
@patch("odoo_data_flow.lib.internal.io.log.error")
def test_write_file_oserror(
    mock_log_error: MagicMock, mock_open: MagicMock, mock_write_csv: MagicMock
) -> None:
    """Test write fle os error.

    Tests that write_file logs an error if an OSError occurs during script writing.
    """
    mock_open.side_effect = OSError("Permission denied on script file")

    write_file(
        filename="data.csv",
        header=["id"],
        data=[["1"]],
        launchfile="protected/load.sh",
        init=True,
        # Provide default worker and batch_size to avoid TypeError
        worker=1,
        batch_size=10,
    )
    mock_write_csv.assert_called_once()
    mock_log_error.assert_called_once()
    assert "Failed to write to launch file" in mock_log_error.call_args[0][0]


@patch("odoo_data_flow.lib.internal.io.log.error")
def test_write_file_invalid_command(mock_log_error: MagicMock) -> None:
    """Tests that an error is logged for an invalid command type."""
    write_file(
        filename="dummy.csv",
        launchfile="dummy.sh",
        command="invalid-command",
    )
    mock_log_error.assert_called_once()
    assert "Invalid command type" in mock_log_error.call_args[0][0]
