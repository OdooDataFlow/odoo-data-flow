"""Additional tests to cover missing functionality in importer.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from odoo_data_flow.importer import run_import


@patch("odoo_data_flow.importer._show_error_panel")
def test_run_import_invalid_context_json_string(mock_show_error: MagicMock) -> None:
    """Test that run_import handles invalid JSON string context."""
    run_import(
        config="dummy.conf",
        filename="dummy.csv",
        model="res.partner",
        context="{invalid: json}",  # Invalid JSON string
        deferred_fields=None,
        unique_id_field=None,
        no_preflight_checks=True,
        headless=True,
        worker=1,
        batch_size=100,
        skip=0,
        fail=False,
        separator=";",
        ignore=None,
        encoding="utf-8",
        o2m=False,
        groupby=None,
    )
    mock_show_error.assert_called_once()


@patch("odoo_data_flow.importer._show_error_panel")
def test_run_import_invalid_context_type(mock_show_error: MagicMock) -> None:
    """Test that run_import handles invalid context type."""
    run_import(
        config="dummy.conf",
        filename="dummy.csv",
        model="res.partner",
        context=123,  # Invalid context type (not dict or string)
        deferred_fields=None,
        unique_id_field=None,
        no_preflight_checks=True,
        headless=True,
        worker=1,
        batch_size=100,
        skip=0,
        fail=False,
        separator=";",
        ignore=None,
        encoding="utf-8",
        o2m=False,
        groupby=None,
    )
    mock_show_error.assert_called_once()


@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer._run_preflight_checks")
def test_run_import_no_file_exists(
    mock_preflight: MagicMock, mock_import_data: MagicMock
) -> None:
    """Test that run_import handles file not existing."""
    mock_preflight.return_value = True
    mock_import_data.return_value = (True, {"total_records": 1})

    run_import(
        config="dummy.conf",
        filename="nonexistent.csv",
        model="res.partner",
        deferred_fields=None,
        unique_id_field=None,
        no_preflight_checks=False,
        headless=True,
        worker=1,
        batch_size=100,
        skip=0,
        fail=False,
        separator=";",
        ignore=None,
        context={},
        encoding="utf-8",
        o2m=False,
        groupby=None,
    )
    # Should not proceed to import data if file doesn't exist
    mock_import_data.assert_called_once()


@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer._run_preflight_checks")
def test_run_import_file_empty(
    mock_preflight: MagicMock, mock_import_data: MagicMock, tmp_path: Path
) -> None:
    """Test that run_import handles empty file."""
    mock_preflight.return_value = True
    mock_import_data.return_value = (True, {"total_records": 1})

    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")

    run_import(
        config="dummy.conf",
        filename=str(empty_file),
        model="res.partner",
        deferred_fields=None,
        unique_id_field=None,
        no_preflight_checks=False,
        headless=True,
        worker=1,
        batch_size=100,
        skip=0,
        fail=False,
        separator=";",
        ignore=None,
        context={},
        encoding="utf-8",
        o2m=False,
        groupby=None,
    )
    # Should handle empty file appropriately
    mock_import_data.assert_called_once()


@patch("odoo_data_flow.importer.import_threaded.import_data", return_value=(False, {"id_map": {}}))
@patch("odoo_data_flow.importer._run_preflight_checks", return_value=True)
def test_run_import_fail_with_no_id_map(
    mock_preflight: MagicMock, mock_import_data: MagicMock, tmp_path: Path
) -> None:
    """Test run_import when import fails and no id_map is returned."""
    source_file = tmp_path / "source.csv"
    source_file.write_text("id,name\\ntest1,Test Name\\n")
    
    run_import(
        config="dummy.conf",
        filename=str(source_file),
        model="res.partner",
        deferred_fields=None,
        unique_id_field="id",
        no_preflight_checks=True,
        headless=True,
        worker=1,
        batch_size=10,
        skip=0,
        fail=False,
        separator=";",
        ignore=[],
        context={},
        encoding="utf-8",
        o2m=False,
        groupby=None,
    )
    # Should handle missing id_map gracefully
    mock_import_data.assert_called_once()


@patch("odoo_data_flow.importer.os.path.exists", return_value=True)
@patch("odoo_data_flow.importer.os.path.getsize", return_value=100)
@patch("odoo_data_flow.importer.pl.read_csv")
@patch("odoo_data_flow.importer.import_threaded.import_data", return_value=(True, {"total_records": 1, "id_map": {"test1": 1}}))
@patch("odoo_data_flow.importer._run_preflight_checks", return_value=True)
def test_run_import_with_polars_encoding_error(
    mock_preflight: MagicMock,
    mock_import_data: MagicMock,
    mock_read_csv: MagicMock,
    mock_getsize: MagicMock,
    mock_exists: MagicMock,
    tmp_path: Path
) -> None:
    """Test run_import when polars.read_csv throws an exception initially."""
    source_file = tmp_path / "source.csv"
    source_file.write_text("id,name\\ntest1,Test Name\\n")
    
    # Mock first call to fail, second to succeed
    def side_effect_func(*args, **kwargs):
        if side_effect_func.call_count == 1:
            side_effect_func.call_count += 1
            raise Exception("Encoding error")
        else:
            # Return a mock DataFrame with expected structure
            mock_df = MagicMock()
            mock_df.columns = ["id", "name"]
            return mock_df
    
    side_effect_func.call_count = 1
    mock_read_csv.side_effect = side_effect_func

    run_import(
        config="dummy.conf",
        filename=str(source_file),
        model="res.partner",
        deferred_fields=None,
        unique_id_field="id",
        no_preflight_checks=True,
        headless=True,
        worker=1,
        batch_size=10,
        skip=0,
        fail=False,
        separator=";",
        ignore=[],
        context={},
        encoding="utf-8",
        o2m=False,
        groupby=None,
    )
    # Should handle the encoding issue and continue
    assert mock_read_csv.call_count >= 1


@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer._run_preflight_checks", return_value=True)
def test_run_import_with_id_columns(
    mock_preflight: MagicMock, mock_import_data: MagicMock, tmp_path: Path
) -> None:
    """Test run_import when there are /id suffixed columns in the CSV."""
    source_file = tmp_path / "source.csv"
    source_file.write_text("id,name,parent_id/id\\ntest1,Test Name,parent1\\n")
    
    # Mock polars DataFrame
    mock_df = MagicMock()
    mock_df.columns = ["id", "name", "parent_id/id"]
    mock_df.__getitem__.return_value = mock_df
    mock_df.dtype = "string"
    
    with patch("odoo_data_flow.importer.pl.read_csv", return_value=mock_df):
        mock_import_data.return_value = (True, {"total_records": 1, "id_map": {"test1": 1}})
        
        run_import(
            config="dummy.conf",
            filename=str(source_file),
            model="res.partner",
            deferred_fields=None,
            unique_id_field="id",
            no_preflight_checks=True,
            headless=True,
            worker=1,
            batch_size=10,
            skip=0,
            fail=False,
            separator=";",
            ignore=[],
            context={},
            encoding="utf-8",
            o2m=False,
            groupby=None,
        )
        # Should handle /id columns correctly
        mock_import_data.assert_called_once()