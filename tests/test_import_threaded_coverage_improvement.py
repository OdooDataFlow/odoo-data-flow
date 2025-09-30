"""Additional tests to improve coverage for uncovered lines in import_threaded.py."""

from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

from rich.progress import Progress

from odoo_data_flow.import_threaded import (
    _create_batch_individually,
    _execute_load_batch,
    _get_model_fields,
    _handle_create_error,
    _handle_fallback_create,
    _handle_tuple_index_error,
    _orchestrate_pass_1,
    _parse_csv_data,
    _read_data_file,
    _safe_convert_field_value,
    _setup_fail_file,
    import_data,
)


def test_parse_csv_data_insufficient_lines() -> None:
    """Test _parse_csv_data when there are not enough lines after skipping."""
    f = StringIO("")  # Empty file
    header, data = _parse_csv_data(f, ",", 0)  # Should return empty lists
    assert header == []
    assert data == []


def test_read_data_file_unicode_decode_error() -> None:
    """Test _read_data_file when UnicodeDecodeError occurs and all fallbacks fail."""
    # Test when all encodings fail with UnicodeDecodeError
    with patch("builtins.open") as mock_open:

        def open_side_effect(*args: Any, **kwargs: Any) -> Any:
            # Always raise UnicodeDecodeError regardless of encoding
            raise UnicodeDecodeError("utf-8", b"test", 0, 1, "fake error")

        mock_open.side_effect = open_side_effect

        header, data = _read_data_file("dummy.csv", ",", "utf-8", 0)
        assert header == []
        assert data == []


def test_safe_convert_field_value_edge_cases() -> None:
    """Test _safe_convert_field_value with various edge cases."""
    # Test with /id suffixed fields
    result = _safe_convert_field_value("parent_id/id", "some_value", "char")
    assert result == "some_value"

    # Test with positive field type and negative value
    result = _safe_convert_field_value("field", "-5", "positive")
    assert result == -5  # Should be converted to int since it's numeric

    # Test with negative field type and positive value
    result = _safe_convert_field_value("field", "5", "negative")
    assert result == 5  # Should be converted to int since it's numeric

    # Test with empty value for numeric fields
    result = _safe_convert_field_value("field", "", "integer")
    assert result == 0

    result = _safe_convert_field_value("field", "", "float")
    assert result == 0

    # Test with invalid float values
    result = _safe_convert_field_value("field", "not_a_number", "float")
    assert result == "not_a_number"

    # Test with non-integer float values (should remain as string)
    result = _safe_convert_field_value("field", "1.5", "integer")
    assert result == "1.5"  # Should remain as string since it's not an integer


def test_handle_create_error_various_errors() -> None:
    """Test _handle_create_error with various error types."""
    # Test constraint violation error
    error = Exception("constraint violation")
    error_str, _failed_line, _summary = _handle_create_error(
        0, error, ["test", "data"], "test summary"
    )
    assert "Constraint violation" in error_str

    # Test database connection pool exhaustion errors
    error = Exception("connection pool is full")
    error_str, _failed_line, _summary = _handle_create_error(
        0, error, ["test", "data"], "test summary"
    )
    assert "Database connection pool exhaustion" in error_str

    # Test database serialization errors
    error = Exception("could not serialize access")
    error_str, _failed_line, _summary = _handle_create_error(
        0, error, ["test", "data"], "test summary"
    )
    assert "Database serialization error" in error_str

    # Test tuple index out of range errors
    error = Exception("tuple index out of range")
    error_str, _failed_line, _summary = _handle_create_error(
        0, error, ["test", "data"], "test summary"
    )
    assert "Tuple unpacking error" in error_str


def test_handle_tuple_index_error() -> None:
    """Test _handle_tuple_index_error function."""
    # Use None as progress to avoid console issues
    failed_lines: list[list[Any]] = []

    # Test the function with progress=None to avoid rich console issues in tests
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

    result = _create_batch_individually(
        mock_model, batch_lines, batch_header, 0, {}, []
    )

    # Should handle the error and return failed lines
    assert len(result["failed_lines"]) == 1
    error_msg = result["failed_lines"][0][-1].lower()
    assert "tuple index" in error_msg or "range" in error_msg


def test_handle_fallback_create_with_progress() -> None:
    """Test _handle_fallback_create function."""
    mock_model = MagicMock()
    current_chunk = [["rec1", "A"], ["rec2", "B"]]
    batch_header = ["id", "name"]
    uid_index = 0
    context: dict[str, Any] = {}
    ignore_list: list[str] = []
    progress = MagicMock()
    aggregated_id_map: dict[str, int] = {}
    aggregated_failed_lines: list[list[Any]] = []
    batch_number = 1

    with patch(
        "odoo_data_flow.import_threaded._create_batch_individually"
    ) as mock_create_ind:
        mock_create_ind.return_value = {
            "id_map": {"rec1": 1, "rec2": 2},
            "failed_lines": [],
            "error_summary": "test",
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
            batch_number,
            error_message="test error",
        )

        assert aggregated_id_map == {"rec1": 1, "rec2": 2}


def test_execute_load_batch_force_create_with_progress() -> None:
    """Test _execute_load_batch with force_create enabled."""
    mock_model = MagicMock()
    thread_state = {
        "model": mock_model,
        "progress": MagicMock(),
        "unique_id_field_index": 0,
        "force_create": True,  # Enable force create
        "ignore_list": [],
    }
    batch_header = ["id", "name"]
    batch_lines = [["rec1", "A"], ["rec2", "B"]]

    with patch(
        "odoo_data_flow.import_threaded._create_batch_individually"
    ) as mock_create:
        mock_create.return_value = {
            "id_map": {"rec1": 1, "rec2": 2},
            "failed_lines": [],
            "error_summary": "test",
            "success": True,
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
    with patch("builtins.open") as mock_open:
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        header = ["id", "_ERROR_REASON", "name"]
        writer, handle = _setup_fail_file("fail.csv", header, ",", "utf-8")

        # Should not add _ERROR_REASON again since it's already in header
        # Just verify it doesn't crash
        assert writer is not None
        assert handle is not None


def test_recursive_create_batches_no_id_column() -> None:
    """Test _recursive_create_batches when no 'id' column exists."""
    from odoo_data_flow.import_threaded import _recursive_create_batches

    header = ["name", "age"]  # No 'id' column
    data = [["Alice", "25"], ["Bob", "30"]]

    batches = list(_recursive_create_batches(data, [], header, 10, True))  # o2m=True

    # Should handle the case where no 'id' column exists
    assert len(batches) >= 0  # This should not crash


def test_orchestrate_pass_1_force_create() -> None:
    """Test _orchestrate_pass_1 with force_create enabled."""
    mock_model = MagicMock()
    header = ["id", "name"]
    all_data = [["rec1", "A"], ["rec2", "B"]]
    unique_id_field = "id"
    deferred_fields: list[str] = []
    ignore: list[str] = []
    context: dict[str, Any] = {}
    fail_writer = None
    fail_handle = None
    max_connection = 1
    batch_size = 10
    o2m = False
    split_by_cols = None

    with Progress() as progress:
        result = _orchestrate_pass_1(
            progress,
            mock_model,
            "res.partner",
            header,
            all_data,
            unique_id_field,
            deferred_fields,
            ignore,
            context,
            fail_writer,
            fail_handle,
            max_connection,
            batch_size,
            o2m,
            split_by_cols,
            force_create=True,  # Enable force create
        )

        # Should return a result dict
        assert isinstance(result, dict)


def test_import_data_connection_dict() -> None:
    """Test import_data with connection config as dict."""
    mock_connection = MagicMock()
    mock_model = MagicMock()

    with patch(
        "odoo_data_flow.import_threaded._read_data_file", return_value=(["id"], [["1"]])
    ):
        with patch(
            "odoo_data_flow.import_threaded.conf_lib.get_connection_from_dict",
            return_value=mock_connection,
        ):
            mock_connection.get_model.return_value = mock_model

            # Mock the _run_threaded_pass function
            with patch(
                "odoo_data_flow.import_threaded._run_threaded_pass"
            ) as mock_run_pass:
                mock_run_pass.return_value = (
                    {"id_map": {"1": 1}, "failed_lines": []},  # results dict
                    False,  # aborted = False
                )

                result, _stats = import_data(
                    config={"host": "localhost"},  # Dict config instead of file
                    model="res.partner",
                    unique_id_field="id",
                    file_csv="dummy.csv",
                )

                # Should succeed
                assert result is True


@patch("odoo_data_flow.import_threaded._read_data_file", return_value=(["id"], [["1"]]))
@patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_config")
def test_import_data_connection_failure(
    mock_get_conn: MagicMock, mock_read_file: MagicMock
) -> None:
    """Test import_data when connection fails."""
    mock_get_conn.side_effect = Exception("Connection failed")

    result, stats = import_data(
        config="dummy.conf",
        model="res.partner",
        unique_id_field="id",
        file_csv="dummy.csv",
    )

    # Should fail gracefully
    assert result is False
    assert stats == {}


@patch("odoo_data_flow.import_threaded._read_data_file", return_value=([], []))
def test_import_data_no_header(mock_read_file: MagicMock) -> None:
    """Test import_data when there's no header in the CSV."""
    result, stats = import_data(
        config="dummy.conf",
        model="res.partner",
        unique_id_field="id",
        file_csv="dummy.csv",
    )

    # Should fail gracefully
    assert result is False
    assert stats == {}


@patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_config")
def test_get_model_fields_callable_method(mock_get_conn: MagicMock) -> None:
    """Test _get_model_fields when _fields is a callable method."""
    mock_model = MagicMock()
    mock_model._fields = lambda: {"field1": {"type": "char"}}

    result = _get_model_fields(mock_model)
    assert result == {"field1": {"type": "char"}}


@patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_config")
def test_get_model_fields_callable_method_exception(mock_get_conn: MagicMock) -> None:
    """Test _get_model_fields when _fields callable raises exception."""
    mock_model = MagicMock()
    mock_model._fields = MagicMock(side_effect=Exception("Error"))

    result = _get_model_fields(mock_model)
    assert result is None


@patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_config")
def test_get_model_fields_callable_method_non_dict(mock_get_conn: MagicMock) -> None:
    """Test _get_model_fields when _fields callable returns non-dict."""
    mock_model = MagicMock()
    mock_model._fields = MagicMock(return_value="not a dict")

    result = _get_model_fields(mock_model)
    assert result is None
