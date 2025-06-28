"""Test the high-level import orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from odoo_data_flow.importer import run_import, run_import_for_migration


@patch("odoo_data_flow.importer.import_threaded.import_data")
def test_run_import_infers_model_from_filename(
    mock_import_data: MagicMock, tmp_path: Path
) -> None:
    """Test infer model name.

    Tests that the model name is correctly inferred from the filename
    when the --model argument is not provided.
    """
    # 1. Setup: Create a dummy file for the function to read.
    source_file = tmp_path / "res_partner.csv"
    source_file.write_text("id,name\n1,test")

    # 2. Action
    run_import(config="dummy.conf", filename=str(source_file), separator=",")

    # 3. Assertions
    mock_import_data.assert_called_once()
    # The second positional argument passed to import_data should be the model name.
    called_model = mock_import_data.call_args.args[1]
    assert called_model == "res.partner"


@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer.log.error")
def test_run_import_no_model_fails(
    mock_log_error: MagicMock, mock_import_data: MagicMock, tmp_path: Path
) -> None:
    """Tests that the import fails if no model can be inferred from the filename."""
    # 1. Setup: A filename starting with a dot will result in an invalid model name.
    bad_file = tmp_path / ".badfilename"
    bad_file.touch()

    # 2. Action
    run_import(config="dummy.conf", filename=str(bad_file))

    # 3. Assertions
    mock_log_error.assert_called_once()
    assert "could not be inferred" in mock_log_error.call_args[0][0]
    # Ensure the import process was stopped and the threaded import was not called
    mock_import_data.assert_not_called()


@patch("odoo_data_flow.importer.import_threaded.import_data")
def test_run_import_fail_mode(mock_import_data: MagicMock, tmp_path: Path) -> None:
    """Test import in fail mode.

    Tests that when --fail is True, the correct parameters for a fail run
    are passed down to the core import function.
    """
    # 1. Setup
    source_file = tmp_path / "res_partner.csv"
    source_file.touch()  # Ensure the source file exists

    # 2. Action
    run_import(
        config="dummy.conf",
        filename=str(source_file),
        model="res.partner",
        fail=True,
    )

    # 3. Assertions
    mock_import_data.assert_called_once()
    call_kwargs = mock_import_data.call_args.kwargs

    # Check that the file paths and flags are set correctly for a fail run
    assert call_kwargs["file_csv"].endswith("res.partner.fail.csv")
    assert call_kwargs["fail_file"].endswith("_failed.csv")
    assert call_kwargs["is_fail_run"] is True
    assert call_kwargs["batch_size"] == 1
    assert call_kwargs["max_connection"] == 1


@patch("odoo_data_flow.importer.log.error")
def test_run_import_bad_context_string(
    mock_log_error: MagicMock, tmp_path: Path
) -> None:
    """Tests that a malformed context string is handled gracefully."""
    # Setup: Create a dummy file to get past the file-read stage
    source_file = tmp_path / "data.csv"
    source_file.write_text("id,name\n1,test")

    run_import(
        config="dummy.conf",
        filename=str(source_file),
        model="res.partner",
        context="this-is-not-a-dict",
    )
    mock_log_error.assert_called_once()
    assert "Invalid context provided" in mock_log_error.call_args[0][0]


@patch("odoo_data_flow.importer.import_threaded.import_data")
def test_run_import_for_migration(mock_import_data: MagicMock) -> None:
    """Tests the in-memory import runner used for migrations."""
    # 1. Action
    run_import_for_migration(
        config="dummy.conf",
        model="res.partner",
        header=["id", "name"],
        data=[["1", "Test"]],
        worker=2,
        batch_size=50,
    )

    # 2. Assertions
    mock_import_data.assert_called_once()
    call_kwargs = mock_import_data.call_args.kwargs
    assert call_kwargs["max_connection"] == 2
    assert call_kwargs["batch_size"] == 50
    assert "tracking_disable" in call_kwargs["context"]
