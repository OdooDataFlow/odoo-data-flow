"""Additional tests to cover missing functionality in import_threaded.py."""

from unittest.mock import MagicMock, patch

from rich.progress import Progress

from odoo_data_flow.import_threaded import (
    _create_batch_individually,
    _execute_load_batch,
    _get_model_fields,
    _handle_create_error,
    _handle_fallback_create,
    _orchestrate_pass_1,
    _parse_csv_data,
    _read_data_file,
    _safe_convert_field_value,
    _setup_fail_file,
    import_data,
)


def test_get_model_fields_none() -> None:
    """Test _get_model_fields when model has no _fields attribute."""
    mock_model = MagicMock()
    delattr(mock_model, "_fields")
    result = _get_model_fields(mock_model)
    assert result is None


def test_get_model_fields_callable_error() -> None:
    """Test _get_model_fields when _fields is callable but raises exception."""
    mock_model = MagicMock()
    mock_model._fields = MagicMock(side_effect=Exception("Error"))
    result = _get_model_fields(mock_model)
    assert result is None


def test_get_model_fields_not_dict() -> None:
    """Test _get_model_fields when _fields returns non-dict."""
    mock_model = MagicMock()
    mock_model._fields = MagicMock(return_value="not a dict")
    result = _get_model_fields(mock_model)
    assert result is None


def test_safe_convert_field_value_with_id_suffix() -> None:
    """Test _safe_convert_field_value with /id suffixed fields."""
    result = _safe_convert_field_value("parent_id/id", "some_value", "char")
    assert result == "some_value"


def test_safe_convert_field_value_integer_positive() -> None:
    """Test _safe_convert_field_value with positive field type."""
    result = _safe_convert_field_value("test_field", "5.0", "positive")
    assert result == 5


def test_safe_convert_field_value_negative() -> None:
    """Test _safe_convert_field_value with negative field type."""
    result = _safe_convert_field_value("test_field", "-5.0", "negative")
    assert result == -5


def test_safe_convert_field_value_float_invalid() -> None:
    """Test _safe_convert_field_value with invalid float."""
    result = _safe_convert_field_value("test_field", "not_a_number", "float")
    assert result == "not_a_number"


def test_handle_create_error_constraint_violation() -> None:
    """Test _handle_create_error with constraint violation error."""
    error = Exception("constraint violation")
    error_str, failed_line, summary = _handle_create_error(
        0, error, ["test", "data"], "test summary"
    )
    assert "Constraint violation" in error_str


def test_handle_create_error_database_pool() -> None:
    """Test _handle_create_error with database connection pool error."""
    error = Exception("connection pool is full")
    error_str, failed_line, summary = _handle_create_error(
        0, error, ["test", "data"], "test summary"
    )
    assert "Database connection pool exhaustion" in error_str


def test_handle_create_error_serialization() -> None:
    """Test _handle_create_error with serialization error."""
    error = Exception("could not serialize access")
    error_str, failed_line, summary = _handle_create_error(
        0, error, ["test", "data"], "test summary"
    )
    assert "Database serialization error" in error_str


def test_parse_csv_data_insufficient_lines() -> None:
    """Test _parse_csv_data when there are not enough lines after skipping."""
    from io import StringIO

    f = StringIO("")  # Empty file
    header, data = _parse_csv_data(f, ",", 0)  # Should return empty lists
    assert header == []
    assert data == []


def test_read_data_file_unicode_decode_error() -> None:
    """Test _read_data_file with UnicodeDecodeError followed by success."""
    with patch("builtins.open") as mock_open:
        # Set up the side effect to raise UnicodeDecodeError for the first attempt with the specified encoding
        # then succeed on a fallback encoding
        file_obj = MagicMock()
        file_obj.__enter__.return_value = MagicMock()  # This would be the file object
        file_obj.__exit__.return_value = False

        # The _read_data_file function first tries with the provided encoding,
        # then falls back to other encodings. We'll mock this process.
        def open_side_effect(*args, **kwargs):
            encoding = kwargs.get("encoding", "utf-8")
            if encoding == "utf-8":
                raise UnicodeDecodeError("utf-8", b"test", 0, 1, "fake error")
            else:
                # For fallback encodings, return the file object
                return file_obj

        mock_open.side_effect = open_side_effect

        # Mock _parse_csv_data to return valid data
        with patch(
            "odoo_data_flow.import_threaded._parse_csv_data",
            return_value=(["id"], [["test"]]),
        ):
            header, data = _read_data_file("dummy.csv", ",", "utf-8", 0)
            # Should have processed with fallback encoding
            assert header == ["id"]


@patch("odoo_data_flow.import_threaded.csv.writer")
def test_setup_fail_file_os_error(mock_csv_writer: MagicMock) -> None:
    """Test _setup_fail_file with OSError."""
    mock_csv_writer.side_effect = OSError("Permission denied")

    with patch("builtins.open", side_effect=OSError("Permission denied")):
        writer, handle = _setup_fail_file("fail.csv", ["id"], ",", "utf-8")
        assert writer is None
        assert handle is None


def test_create_batch_individually_tuple_index_error() -> None:
    """Test _create_batch_individually with tuple index out of range error."""
    mock_model = MagicMock()
    mock_model.browse().env.ref.return_value = None  # No existing record

    # Mock the create method to raise tuple index error
    mock_model.create.side_effect = IndexError("tuple index out of range")

    batch_header = ["id", "name"]
    batch_lines = [["test1", "Test Name"]]

    result = _create_batch_individually(
        mock_model, batch_lines, batch_header, 0, {}, []
    )

    # Should handle the error and return failed lines
    assert len(result["failed_lines"]) == 1
    error_msg = result["failed_lines"][0][-1].lower()
    # Check for the expected error messages
    assert "tuple index" in error_msg or "out of range" in error_msg


class TestExecuteLoadBatchEdgeCases:
    """Additional tests for _execute_load_batch edge cases."""

    @patch("odoo_data_flow.import_threaded._create_batch_individually")
    def test_execute_load_batch_force_create(
        self, mock_create_individually: MagicMock
    ) -> None:
        """Test _execute_load_batch with force_create enabled."""
        mock_model = MagicMock()
        mock_progress = MagicMock()
        thread_state = {
            "model": mock_model,
            "progress": mock_progress,
            "unique_id_field_index": 0,
            "force_create": True,  # Enable force create
            "ignore_list": [],
        }
        batch_header = ["id", "name"]
        batch_lines = [["rec1", "A"], ["rec2", "B"]]

        mock_create_individually.return_value = {
            "id_map": {"rec1": 1, "rec2": 2},
            "failed_lines": [],
            "error_summary": "test",
        }

        result = _execute_load_batch(thread_state, batch_lines, batch_header, 1)

        assert result["success"] is True
        assert result["id_map"] == {"rec1": 1, "rec2": 2}
        mock_create_individually.assert_called_once()

    @patch("odoo_data_flow.import_threaded._create_batch_individually")
    def test_execute_load_batch_serialization_retry_limit(
        self, mock_create_individually: MagicMock
    ) -> None:
        """Test _execute_load_batch with serialization retry limit."""
        mock_model = MagicMock()
        mock_model.load.side_effect = Exception("could not serialize access")
        mock_progress = MagicMock()
        thread_state = {
            "model": mock_model,
            "progress": mock_progress,
            "unique_id_field_index": 0,
            "ignore_list": [],
        }
        batch_header = ["id", "name"]
        batch_lines = [["rec1", "A"], ["rec2", "B"]]

        mock_create_individually.return_value = {
            "id_map": {"rec1": 1, "rec2": 2},
            "failed_lines": [],
            "error_summary": "test",
        }

        result = _execute_load_batch(thread_state, batch_lines, batch_header, 1)

        # Should eventually fall back to create individual - called once per record if load fails
        assert result["success"] is True
        assert mock_create_individually.call_count >= 1


def test_handle_fallback_create() -> None:
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

    # Mock the _create_batch_individually function
    with patch(
        "odoo_data_flow.import_threaded._create_batch_individually"
    ) as mock_individual:
        mock_individual.return_value = {
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

        # Should update the aggregated results
        assert aggregated_id_map == {"rec1": 1, "rec2": 2}


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

                result, stats = import_data(
                    config={"host": "localhost"},  # Dict config instead of file
                    model="res.partner",
                    unique_id_field="id",
                    file_csv="dummy.csv",
                )

                # Should succeed
                assert result is True
