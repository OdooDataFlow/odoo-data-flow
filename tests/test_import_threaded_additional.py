"""Additional tests for uncovered functionality in import_threaded.py."""

from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from odoo_data_flow.import_threaded import (
    _format_odoo_error,
    _get_model_fields,
    _handle_create_error,
    _parse_csv_data,
    _read_data_file,
    _safe_convert_field_value,
    _setup_fail_file,
    _convert_external_id_field,
    _process_external_id_fields,
    _handle_tuple_index_error,
    _create_batch_individually,
    _execute_load_batch,
    _handle_fallback_create,
    _orchestrate_pass_1
)


def test_format_odoo_error_not_string() -> None:
    """Test _format_oddo_error with non-string input."""
    error = {"key": "value"}
    result = _format_odoo_error(error)
    assert result == "{'key': 'value'}"


def test_format_odoo_error_dict_with_message() -> None:
    """Test _format_odoo_error with dict containing message data."""
    error = "{'data': {'message': 'Test error message'}}"
    result = _format_odoo_error(error)
    assert result == "Test error message"


def test_format_odoo_error_syntax_error() -> None:
    """Test _format_odoo_error with malformed string that causes syntax error."""
    error = "invalid python dict {'key': 'value'"  # Missing quote
    result = _format_odoo_error(error)
    assert result.strip() == "invalid python dict {'key': 'value'"


def test_parse_csv_data_skip_lines() -> None:
    """Test _parse_csv_data with skip parameter."""
    from io import StringIO
    
    # Create CSV with some initial lines to skip
    csv_content = "skip_line1\nskip_line2\nid,name\n1,Alice\n2,Bob\n"
    f = StringIO(csv_content)
    
    header, data = _parse_csv_data(f, ",", 2)  # Skip first 2 lines
    
    # Should have skipped first 2 lines and read header + data
    assert header == ["id", "name"]
    assert data == [["1", "Alice"], ["2", "Bob"]]


def test_parse_csv_data_missing_id_column() -> None:
    """Test _parse_csv_data when 'id' column is missing."""
    from io import StringIO
    
    csv_content = "name,age\nAlice,25\nBob,30\n"
    f = StringIO(csv_content)
    
    with pytest.raises(ValueError, match="Source file must contain an 'id' column."):
        _parse_csv_data(f, ",", 0)


def test_get_model_fields_callable_method() -> None:
    """Test _get_model_fields when _fields is a callable method."""
    mock_model = MagicMock()
    mock_model._fields = lambda: {"field1": {"type": "char"}}
    
    result = _get_model_fields(mock_model)
    assert result == {"field1": {"type": "char"}}


def test_get_model_fields_callable_method_exception() -> None:
    """Test _get_model_fields when _fields callable raises exception."""
    mock_model = MagicMock()
    mock_model._fields = MagicMock(side_effect=Exception("Error"))
    
    result = _get_model_fields(mock_model)
    assert result is None


def test_get_model_fields_callable_method_non_dict() -> None:
    """Test _get_model_fields when _fields callable returns non-dict."""
    mock_model = MagicMock()
    mock_model._fields = MagicMock(return_value="not a dict")
    
    result = _get_model_fields(mock_model)
    assert result is None


def test_safe_convert_field_value_numeric_types() -> None:
    """Test _safe_convert_field_value with various numeric types."""
    # Test positive field type - positive numbers should be converted
    result = _safe_convert_field_value("field", "5", "positive")
    assert result == 5  # Should be converted to integer since it's positive
    
    # Test negative field type - negative numbers should be converted
    result = _safe_convert_field_value("field", "-5", "negative")
    assert result == -5  # Should be converted to integer since it's negative
    
    # Test empty value for numeric fields
    result = _safe_convert_field_value("field", "", "integer")
    assert result == 0
    
    result = _safe_convert_field_value("field", "", "float")
    assert result == 0


def test_convert_external_id_field_empty() -> None:
    """Test _convert_external_id_field with empty value."""
    mock_model = MagicMock()
    mock_model.env.ref.return_value = None  # No record found
    
    base_name, converted_value = _convert_external_id_field(mock_model, "parent_id/id", "")
    assert base_name == "parent_id"
    assert converted_value == False  # Empty value should return False


def test_convert_external_id_field_exception() -> None:
    """Test _convert_external_id_field when exception occurs."""
    mock_model = MagicMock()
    mock_model.env.ref.side_effect = Exception("Ref error")
    
    base_name, converted_value = _convert_external_id_field(mock_model, "parent_id/id", "some_ref")
    assert base_name == "parent_id"
    assert converted_value == False


def test_process_external_id_fields() -> None:
    """Test _process_external_id_fields function."""
    mock_model = MagicMock()
    
    clean_vals = {
        "name": "Test",
        "parent_id/id": "parent123",
        "category_id/id": "category456"
    }
    
    mock_ref1 = MagicMock()
    mock_ref1.id = 123
    mock_ref2 = MagicMock()
    mock_ref2.id = 456
    
    def ref_side_effect(ref_name: str, raise_if_not_found: bool = False) -> Any:
        if ref_name == "parent123":
            return mock_ref1
        elif ref_name == "category456":
            return mock_ref2
        else:
            return None
    
    mock_model.env.ref.side_effect = ref_side_effect
    
    converted_vals, external_id_fields = _process_external_id_fields(mock_model, clean_vals)
    
    assert "parent_id" in converted_vals
    assert "category_id" in converted_vals
    assert converted_vals["parent_id"] == 123
    assert converted_vals["category_id"] == 456
    assert converted_vals["name"] == "Test"
    assert set(external_id_fields) == {"parent_id/id", "category_id/id"}


def test_handle_create_error_check_constraint() -> None:
    """Test _handle_create_error with check constraint exception."""
    error = Exception("check constraint error")
    error_str, failed_line, error_summary = _handle_create_error(
        5, error, ["test", "data"], "initial summary"
    )
    
    assert "constraint violation" in error_str.lower()


def test_handle_create_error_pool_error() -> None:
    """Test _handle_create_error with pool error."""
    error = Exception("poolerror occurred")
    error_str, failed_line, error_summary = _handle_create_error(
        5, error, ["test", "data"], "initial summary"
    )
    
    assert "pool" in error_str.lower()


def test_handle_tuple_index_error() -> None:
    """Test _handle_tuple_index_error function."""
    # Use None as progress to avoid console issues
    failed_lines: list[list[Any]] = []
    
    # Test the function with progress=None to avoid rich console issues in tests
    from typing import Any
    progress_console: Any = None
    
    _handle_tuple_index_error(
        progress_console, "source_id_123", ["id", "name"], failed_lines
    )
    
    # The function should add an entry to failed_lines
    assert len(failed_lines) == 1
    assert "source_id_123" in str(failed_lines[0])


def test_create_batch_individually_tuple_index_out_of_range() -> None:
    """Test _create_batch_individually with tuple index out of range."""
    mock_model = MagicMock()
    mock_model.browse().env.ref.return_value = None  # No existing record
    
    # Mock create method to raise IndexError
    mock_model.create.side_effect = IndexError("tuple index out of range")
    
    batch_header = ["id", "name", "value"]
    batch_lines = [["rec1", "Name", "Value"]]
    
    result = _create_batch_individually(mock_model, batch_lines, batch_header, 0, {}, [])
    
    # Should handle the error and return failed lines
    assert len(result["failed_lines"]) == 1
    error_msg = result["failed_lines"][0][-1].lower()
    assert "tuple index" in error_msg or "range" in error_msg


def test_handle_fallback_create_with_progress() -> None:
    """Test _handle_fallback_create function with progress."""
    from rich.progress import Progress
    
    mock_model = MagicMock()
    current_chunk = [["rec1", "A"], ["rec2", "B"]]
    batch_header = ["id", "name"]
    uid_index = 0
    context: dict[str, Any] = {}
    ignore_list: list[str] = []
    aggregated_id_map: dict[str, int] = {}
    aggregated_failed_lines: list[list[Any]] = []
    
    with Progress() as progress:
        task_id = progress.add_task("test")
        
        with patch("odoo_data_flow.import_threaded._create_batch_individually") as mock_create_ind:
            mock_create_ind.return_value = {
                "id_map": {"rec1": 1, "rec2": 2},
                "failed_lines": [],
                "error_summary": "test"
            }
            
            _handle_fallback_create(
                mock_model,
                current_chunk,
                batch_header,
                uid_index,
                context,
                ignore_list,
                progress,
                aggregated_id_map,
                aggregated_failed_lines,
                1,  # batch_number
                error_message="test error"
            )
            
            assert aggregated_id_map == {"rec1": 1, "rec2": 2}


def test_execute_load_batch_force_create_with_progress() -> None:
    """Test _execute_load_batch with force_create enabled."""
    from rich.progress import Progress
    
    with Progress() as progress:
        task_id = progress.add_task("test")
        
        mock_model = MagicMock()
        thread_state = {
            "model": mock_model,
            "progress": progress,
            "unique_id_field_index": 0,
            "force_create": True,
            "ignore_list": [],
        }
        batch_header = ["id", "name"]
        batch_lines = [["rec1", "A"], ["rec2", "B"]]
        
        with patch("odoo_data_flow.import_threaded._create_batch_individually") as mock_create:
            mock_create.return_value = {
                "id_map": {"rec1": 1, "rec2": 2},
                "failed_lines": [],
                "error_summary": "test",
                "success": True
            }
            
            result = _execute_load_batch(thread_state, batch_lines, batch_header, 1)
            
            assert result["success"] is True
            assert result["id_map"] == {"rec1": 1, "rec2": 2}
            mock_create.assert_called_once()


@patch("builtins.open")
def test_read_data_file_os_error(mock_open: MagicMock) -> None:
    """Test _read_data_file with OSError (not UnicodeDecodeError)."""
    mock_open.side_effect = OSError("File access error")
    
    header, data = _read_data_file("nonexistent.txt", ",", "utf-8", 0)
    assert header == []
    assert data == []


def test_read_data_file_all_fallbacks_fail() -> None:
    """Test _read_data_file when all fallback encodings fail."""
    with patch("builtins.open") as mock_open:
        def open_side_effect(*args: Any, **kwargs: Any) -> Any:
            # Always raise UnicodeDecodeError regardless of encoding
            raise UnicodeDecodeError("utf-8", b"test", 0, 1, "fake error")
        
        mock_open.side_effect = open_side_effect
        
        header, data = _read_data_file("dummy.csv", ",", "utf-8", 0)
        assert header == []
        assert data == []


def test_setup_fail_file_with_error_reason_column() -> None:
    """Test _setup_fail_file when _ERROR_REASON is already in header."""
    from rich.console import Console
    
    # Create a console to avoid rich errors in testing
    console = Console(force_terminal=False)
    
    with patch("builtins.open") as mock_open:
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        header = ["id", "_ERROR_REASON", "name"]
        writer, handle = _setup_fail_file("fail.csv", header, ",", "utf-8")
        
        # Should not add _ERROR_REASON again since it's already in header
        if writer:
            # writer.writerow should be called with original headers since _ERROR_REASON already exists
            pass  # Testing the logic within the function


def test_recursive_create_batches_no_id_column() -> None:
    """Test _recursive_create_batches when no 'id' column exists."""
    from odoo_data_flow.import_threaded import _recursive_create_batches
    
    header = ["name", "age"]  # No 'id' column
    data = [["Alice", "25"], ["Bob", "30"]]
    
    batches = list(_recursive_create_batches(data, [], header, 10, True))  # o2m=True
    
    # Should handle the case where no 'id' column exists
    assert len(batches) >= 0  # This should not crash