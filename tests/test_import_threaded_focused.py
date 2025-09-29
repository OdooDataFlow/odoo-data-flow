"""Focused tests for import_threaded to improve coverage."""

import io
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from odoo_data_flow.import_threaded import (
    _convert_external_id_field,
    _create_batch_individually,
    _filter_ignored_columns,
    _format_odoo_error,
    _get_model_fields,
    _handle_create_error,
    _parse_csv_data,
    _prepare_pass_2_data,
    _read_data_file,
    _recursive_create_batches,
    _setup_fail_file,
)


class TestFormatOdooError:
    """Test _format_odoo_error function."""

    def test_format_odoo_error_not_string(self) -> None:
        """Test when error is not a string."""
        result = _format_odoo_error(123)
        assert result == "123"

    def test_format_odoo_error_non_parsable_string(self) -> None:
        """Test when error is a non-parsable string."""
        result = _format_odoo_error("Just a string")
        assert result == "Just a string"

    def test_format_odoo_error_parsable_with_message(self) -> None:
        """Test when error is a parsable dict with message."""
        error = "{'data': {'message': 'Test error message'}}"
        result = _format_odoo_error(error)
        assert result == "Test error message"

    def test_format_odoo_error_parsable_fallback(self) -> None:
        """Test when error dict doesn't have the expected structure."""
        error = "{'other': 'data'}"
        result = _format_odoo_error(error)
        assert result == "{'other': 'data'}"


class TestParseCSVData:
    """Test _parse_csv_data function."""

    def test_parse_csv_data_simple(self) -> None:
        """Test parsing basic CSV data."""
        # Create a string buffer to simulate a file
        csv_content = "id;name\n1;Test\n2;Another"
        f = io.StringIO(csv_content)
        header, data = _parse_csv_data(f, ";", 0)
        assert len(data) == 2
        assert data[0][0] == "1"  # id
        assert data[0][1] == "Test"  # name

    def test_parse_csv_data_empty(self) -> None:
        """Test parsing empty CSV data."""
        f = io.StringIO("")
        header, data = _parse_csv_data(f, ";", 0)
        assert data == []
        assert header == []


class TestReadDataFile:
    """Test _read_data_file function."""

    def test_read_data_file_success(self) -> None:
        """Test reading CSV file successfully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id;name\n1;Test\n2;Another\n")
            f.flush()
            filepath = f.name

        header, data = _read_data_file(filepath, ";", "utf-8", 0)
        assert len(data) == 2
        assert data[0][0] == "1"  # id
        assert data[0][1] == "Test"  # name
        import os

        os.unlink(filepath)

    def test_read_data_file_not_found(self) -> None:
        """Test reading non-existent file."""
        header, data = _read_data_file("/nonexistent.csv", ";", "utf-8", 0)
        assert data == []
        assert header == []

    def test_read_data_file_no_id_column(self) -> None:
        """Test reading file without id column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name;value\nTest;1\nAnother;2\n")
            f.flush()
            filepath = f.name

        # This should raise ValueError, catch it and check
        with pytest.raises(ValueError, match="Source file must contain an 'id' column"):
            header, _data = _read_data_file(filepath, ";", "utf-8", 0)
        import os

        os.unlink(filepath)


class TestFilterIgnoredColumns:
    """Test _filter_ignored_columns function."""

    def test_filter_ignored_columns(self) -> None:
        """Test filtering ignored columns."""
        header = ["id", "name", "to_ignore", "value"]
        data = [
            ["1", "Test", "ignore_value", "val1"],
            ["2", "Test2", "ignore_value2", "val2"],
        ]
        ignore_list = ["to_ignore"]
        new_header, new_data = _filter_ignored_columns(ignore_list, header, data)
        assert "to_ignore" not in new_header
        assert "id" in new_header
        assert "name" in new_header
        # Each row should have the ignored column removed
        assert len(new_data[0]) == 3  # id, name, value

    def test_filter_ignored_columns_none(self) -> None:
        """Test filtering with empty ignored list."""
        header = ["id", "name", "value"]
        data = [["1", "Test", "val1"], ["2", "Test2", "val2"]]
        ignore_list: list[str] = []
        new_header, new_data = _filter_ignored_columns(ignore_list, header, data)
        assert new_header == header
        assert new_data == data


class TestSetupFailFile:
    """Test _setup_fail_file function."""

    def test_setup_fail_file_success(self) -> None:
        """Test setting up fail file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fail_filename = Path(tmpdir) / "fail.csv"
            writer, handle = _setup_fail_file(
                str(fail_filename), ["id", "name"], ";", "utf-8"
            )
            assert writer is not None
            assert handle is not None
            handle.close()

    def test_setup_fail_file_os_error(self) -> None:
        """Test setting up fail file with OS error."""
        writer, handle = _setup_fail_file(
            "/root/nonexistent/fail.csv", ["id", "name"], ";", "utf-8"
        )
        assert writer is None
        assert handle is None


class TestPreparePass2Data:
    """Test _prepare_pass_2_data function."""

    def test_prepare_pass_2_data(self) -> None:
        """Test preparing data for pass 2."""
        all_data = [
            ["1", "ref1", "cat1,cat2"],
            ["2", "ref2", "cat2"],
        ]
        header = ["id", "name/id", "category_ids/id"]
        unique_id_field_index = 0
        id_map = {"1": 100, "2": 200}
        deferred_fields = ["name/id", "category_ids/id"]
        result = _prepare_pass_2_data(
            all_data, header, unique_id_field_index, id_map, deferred_fields
        )

        # Should return list of (db_id, update_vals) tuples
        assert isinstance(result, list)
        if result:
            db_id, update_vals = result[0]
            assert isinstance(db_id, int)
            assert isinstance(update_vals, dict)


class TestRecursiveCreateBatches:
    """Test _recursive_create_batches function."""

    def test_recursive_create_batches_single_column(self) -> None:
        """Test recursive batch creation with single column."""
        current_data = [
            ["1", "Test", "A"],
            ["2", "Test2", "B"],
        ]
        header = ["id", "name", "tags"]
        group_cols: list[str] = []
        batch_size = 2
        o2m = False
        result = list(
            _recursive_create_batches(current_data, group_cols, header, batch_size, o2m)
        )
        assert len(result) >= 1

    def test_recursive_create_batches_multiple_columns(self) -> None:
        """Test recursive batch creation with multiple columns."""
        current_data = [
            ["1", "Test", "A"],
            ["2", "Test2", "A"],  # Same tag for grouping
            ["3", "Test3", "B"],
        ]
        header = ["id", "name", "tags"]
        group_cols = ["tags"]
        batch_size = 1  # Small batch size to force multiple chunks
        o2m = False
        result = list(
            _recursive_create_batches(current_data, group_cols, header, batch_size, o2m)
        )
        assert len(result) >= 1


class TestGetModelFields:
    """Test _get_model_fields function."""

    @patch("odoo_data_flow.import_threaded.conf_lib")
    def test_get_model_fields_success(self, mock_conf_lib: Mock) -> None:
        """Test getting model fields successfully."""
        mock_model = Mock()
        mock_model._fields = {"id": {"type": "integer"}}

        result = _get_model_fields(mock_model)
        assert result is not None
        assert "id" in result

    @patch("odoo_data_flow.import_threaded.conf_lib")
    def test_get_model_fields_exception(self, mock_conf_lib: Mock) -> None:
        """Test getting model fields with exception."""
        mock_model = Mock()
        del (
            mock_model._fields
        )  # Remove the _fields attribute to trigger the exception path

        result = _get_model_fields(mock_model)
        assert result is None


class TestConvertExternalIdField:
    """Test methods within RPCThreadImport class."""

    def test_convert_external_id_field(self) -> None:
        """Test converting external ID field."""
        # Create a mock model
        mock_model = Mock()
        result = _convert_external_id_field(
            model=mock_model, field_name="parent_id/id", field_value="module.ref1"
        )
        # The function returns a tuple (base_field_name, converted_value)
        base_field_name, _converted_value = result
        assert base_field_name == "parent_id"
        # Since we're mocking, the converted value will depend on the mock behavior

    def test_convert_external_id_field_special_chars(self) -> None:
        """Test converting external ID field with special characters."""
        # Create a mock model
        mock_model = Mock()
        result = _convert_external_id_field(
            model=mock_model,
            field_name="parent_id/id",
            field_value="module.name-with.special/chars",
        )
        # The function returns a tuple (base_field_name, converted_value)
        base_field_name, _converted_value = result
        assert base_field_name == "parent_id"
        # Since we're mocking, the converted value will depend on the mock behavior


class TestHandleCreateError:
    """Test _handle_create_error function."""

    def test_handle_create_error_connection(self) -> None:
        """Test handling create error with connection error."""
        # Mock a connection object and batch
        Mock()

        # Test the function with correct signature
        i = 0
        create_error = Exception("Connection error")
        line = ["1", "test"]
        error_summary = "Initial error"

        # This function has complex signature, test by calling it
        with patch("odoo_data_flow.import_threaded.log"):
            error_message, failed_line, _new_error_summary = _handle_create_error(
                i, create_error, line, error_summary
            )
            assert isinstance(error_message, str)
            assert isinstance(failed_line, list)


class TestCreateBatchIndividually:
    """Test _create_batch_individually function."""

    def test_create_batch_individually_success(self) -> None:
        """Test creating batch individually with success."""
        # Mock objects for the function
        mock_model = Mock()
        batch_lines = [["1", "Test"]]
        batch_header = ["id", "name"]
        uid_index = 0
        context: dict[str, Any] = {}
        ignore_list: list[str] = []

        # Mock the load method to return success
        mock_model.load.return_value = [[1], []]  # Success IDs, errors

        with patch(
            "odoo_data_flow.import_threaded._handle_create_error"
        ) as mock_handle_error:
            mock_handle_error.return_value = {
                "id_map": {},
                "failed_lines": [],
                "connection_failure": False,
            }

            result = _create_batch_individually(
                mock_model, batch_lines, batch_header, uid_index, context, ignore_list
            )
            assert "id_map" in result
            assert "failed_lines" in result
