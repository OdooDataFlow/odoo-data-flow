"""Test the high-level import orchestrator, including pre-flight checks."""

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
    """Test import with no fails.

    Tests that the import fails if no model can be inferred from the filename.
    """
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
    assert call_kwargs["file_csv"].endswith("res_partner_fail.csv")
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
        separator=",",
    )
    mock_log_error.assert_called_once()
    assert "Invalid context provided" in mock_log_error.call_args[0][0]


@patch("odoo_data_flow.importer.log.error")
def test_run_import_context_not_a_dict(
    mock_log_error: MagicMock, tmp_path: Path
) -> None:
    """Tests that an error is logged if the context string is not a dictionary."""
    # Setup: Create a dummy file to get past the file-read stage
    source_file = tmp_path / "data.csv"
    source_file.write_text("id,name\n1,test")

    # This context is a valid Python literal, but it's a list, not a dict.
    run_import(
        config="dummy.conf",
        filename=str(source_file),
        model="res.partner",
        context="['not', 'a', 'dict']",
        separator=",",
    )
    mock_log_error.assert_called_once()
    assert "Context must be a dictionary" in mock_log_error.call_args[0][0]


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


@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer._verify_import_fields")
def test_run_import_skips_verification_by_default(
    mock_verify_fields: MagicMock, mock_import_data: MagicMock, tmp_path: Path
) -> None:
    """Tests that the field verification step is NOT run if the flag is omitted."""
    source_file = tmp_path / "data.csv"
    source_file.write_text("id,name\n1,test")

    run_import(
        config="dummy.conf",
        filename=str(source_file),
        model="res.partner",
        verify_fields=False,  # Explicitly False
    )

    mock_verify_fields.assert_not_called()
    mock_import_data.assert_called_once()


@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer.conf_lib.get_connection_from_config")
def test_verify_fields_success(
    mock_get_connection: MagicMock, mock_import_data: MagicMock, tmp_path: Path
) -> None:
    """Tests the success path where all CSV columns exist on the Odoo model."""
    # 1. Setup
    source_file = tmp_path / "data.csv"
    source_file.write_text("id,name,email")  # Header for the file to be read

    # Mock the Odoo model object and its return value
    mock_model_fields_obj = MagicMock()
    # Simulate Odoo returning a list of valid fields
    mock_model_fields_obj.search_read.return_value = [
        {"name": "id"},
        {"name": "name"},
        {"name": "email"},
    ]
    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_model_fields_obj
    mock_get_connection.return_value = mock_connection

    # 2. Action
    run_import(
        config="dummy.conf",
        filename=str(source_file),
        model="res.partner",
        verify_fields=True,
        separator=",",  # Use the correct separator for the test file
    )

    # 3. Assertions
    # The verification should pass, and the main import function should be called
    mock_import_data.assert_called_once()


@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer.log.error")
@patch("odoo_data_flow.importer.conf_lib.get_connection_from_config")
def test_verify_fields_failure_missing_field(
    mock_get_connection: MagicMock,
    mock_log_error: MagicMock,
    mock_import_data: MagicMock,
    tmp_path: Path,
) -> None:
    """Tests the failure path where a CSV column does not exist on the Odoo model."""
    # 1. Setup
    source_file = tmp_path / "data.csv"
    # This file contains a column that is not on the mocked model below
    source_file.write_text("id,name,x_studio_legacy_field")

    mock_model_fields_obj = MagicMock()
    # Simulate Odoo returning only two valid fields
    mock_model_fields_obj.search_read.return_value = [
        {"name": "id"},
        {"name": "name"},
    ]
    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_model_fields_obj
    mock_get_connection.return_value = mock_connection

    # 2. Action
    run_import(
        config="dummy.conf",
        filename=str(source_file),
        model="res.partner",
        verify_fields=True,
        separator=",",
    )

    # 3. Assertions
    # An error should be logged, and the main import should NOT be called
    assert mock_log_error.call_count > 0
    # Check that the specific error message was one of the logs
    assert any(
        "is not a valid field" in call[0][0] for call in mock_log_error.call_args_list
    )
    mock_import_data.assert_not_called()


@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer.log.error")
def test_verify_fields_failure_file_not_found(
    mock_log_error: MagicMock, mock_import_data: MagicMock
) -> None:
    """Tests that verification fails gracefully if the source file is not found."""
    run_import(
        config="dummy.conf",
        filename="non_existent_file.csv",
        model="res.partner",
        verify_fields=True,
    )

    # Assert that the specific error message was logged
    assert any(
        "Input file not found" in call[0][0] for call in mock_log_error.call_args_list
    )
    mock_import_data.assert_not_called()
