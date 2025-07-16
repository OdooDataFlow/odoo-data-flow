"""Test the high-level import orchestrator, including pre-flight checks."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from odoo_data_flow.importer import run_import, run_import_for_migration


@patch("odoo_data_flow.importer._run_preflight_checks", return_value=True)
@patch("odoo_data_flow.importer.import_threaded.import_data")
def test_run_import_infers_model_from_filename(
    mock_import_data: MagicMock, mock_run_checks: MagicMock, tmp_path: Path
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
    mock_run_checks.assert_called_once()
    mock_import_data.assert_called_once()
    # The second positional argument passed to import_data should be the model name.
    called_model = mock_import_data.call_args.args[1]
    assert called_model == "res.partner"


@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer._show_error_panel")
def test_run_import_no_model_fails(
    mock_show_error: MagicMock, mock_import_data: MagicMock, tmp_path: Path
) -> None:
    """Test import with no fails.

    Tests that the import fails if no model can be inferred from the filename.
    """
    # 1. Setup: A filename starting with a dot will result in an invalid model name.
    bad_file = tmp_path / ".badfilename"
    bad_file.touch()

    # 2. Action
    run_import(config="dummy.conf", filename=str(bad_file))
    mock_show_error.assert_called_once()
    assert "Model Not Found" in mock_show_error.call_args.args[0]
    mock_import_data.assert_not_called()


@patch("odoo_data_flow.importer._run_preflight_checks", return_value=True)
@patch("odoo_data_flow.importer.import_threaded.import_data")
def test_run_import_fail_mode(
    mock_import_data: MagicMock, mock_run_checks: MagicMock, tmp_path: Path
) -> None:
    """Test import in fail mode.

    Tests that when --fail is True, the correct parameters for a fail run
    are passed down to the core import function.
    """
    # 1. Setup
    source_file = tmp_path / "res_partner.csv"
    source_file.write_text("id,name\n1,test")  # Give the file some content
    (tmp_path / "res_partner_fail.csv").touch()  # The fail file must exist

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


@patch("odoo_data_flow.importer._run_preflight_checks")
@patch("odoo_data_flow.importer.import_threaded.import_data")
def test_preflight_checks_run_by_default(
    mock_import_data: MagicMock, mock_run_checks: MagicMock, tmp_path: Path
) -> None:
    """Tests that the pre-flight checks are run by default."""
    source_file = tmp_path / "data.csv"
    source_file.write_text("id,name\n1,test")
    mock_run_checks.return_value = True  # Simulate checks passing

    run_import(config="dummy.conf", filename=str(source_file), model="res.partner")

    mock_run_checks.assert_called_once()
    mock_import_data.assert_called_once()


@patch("odoo_data_flow.importer._run_preflight_checks", return_value=True)
@patch("odoo_data_flow.importer._show_error_panel")
def test_run_import_bad_context_string(
    mock_show_error: MagicMock, mock_run_checks: MagicMock, tmp_path: Path
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
    mock_show_error.assert_called_once()
    assert "Invalid Context" in mock_show_error.call_args.args[0]


@patch("odoo_data_flow.importer._run_preflight_checks", return_value=True)
@patch("odoo_data_flow.importer._show_error_panel")
def test_run_import_context_not_a_dict(
    mock_show_error: MagicMock, mock_run_checks: MagicMock, tmp_path: Path
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
    mock_show_error.assert_called_once()
    assert "Invalid Context" in mock_show_error.call_args[0]


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


@patch("odoo_data_flow.importer._run_preflight_checks")
@patch("odoo_data_flow.importer.import_threaded.import_data")
def test_preflight_checks_are_skipped(
    mock_import_data: MagicMock, mock_run_checks: MagicMock, tmp_path: Path
) -> None:
    """Tests that the pre-flight checks are skipped when the flag is passed."""
    source_file = tmp_path / "data.csv"
    source_file.write_text("id,name\n1,test")

    run_import(
        config="dummy.conf",
        filename=str(source_file),
        model="res.partner",
        no_preflight_checks=True,
    )
    mock_run_checks.assert_not_called()
    mock_import_data.assert_called_once()


@patch("odoo_data_flow.importer._run_preflight_checks")
@patch("odoo_data_flow.importer.import_threaded.import_data")
def test_import_aborts_on_preflight_failure(
    mock_import_data: MagicMock, mock_run_checks: MagicMock, tmp_path: Path
) -> None:
    """Tests that the import is aborted if a pre-flight check fails."""
    source_file = tmp_path / "data.csv"
    source_file.write_text("id,name\n1,test")
    mock_run_checks.return_value = False  # Simulate a check failing

    run_import(config="dummy.conf", filename=str(source_file), model="res.partner")

    mock_run_checks.assert_called_once()
    mock_import_data.assert_not_called()
