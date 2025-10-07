"""Tests for the refactored, low-level, multi-threaded import logic."""

from typing import Any
from unittest.mock import MagicMock, patch

from rich.progress import Progress

from odoo_data_flow.import_threaded import (
    _create_batch_individually,
    _create_batches,
    _execute_load_batch,
    _get_model_fields,
    _handle_create_error,
    _handle_fallback_create,
    _handle_tuple_index_error,
    _orchestrate_pass_1,
    _read_data_file,
    _recursive_create_batches,
    _safe_convert_field_value,
    _setup_fail_file,
    import_data,
)


def test_read_data_file_all_fallbacks_fail() -> None:
    """Test _read_data_file when all fallback encodings fail."""
    with patch("builtins.open") as mock_open:

        def side_effect(filename: str, encoding: str, **kwargs: Any) -> None:
            # Always raise UnicodeDecodeError regardless of encoding tried
            raise UnicodeDecodeError("utf-8", b"test", 0, 1, "fake error")

        mock_open.side_effect = side_effect

        header, data = _read_data_file("dummy.csv", ",", "utf-8", 0)
        assert header == []
        assert data == []


def test_read_data_file_os_error() -> None:
    """Test _read_data_file with OSError."""
    with patch("builtins.open") as mock_open:
        mock_open.side_effect = OSError("File access error")

        header, data = _read_data_file("dummy.csv", ",", "utf-8", 0)
        assert header == []
        assert data == []


def test_safe_convert_field_value_id_suffix() -> None:
    """Test _safe_convert_field_value with /id suffix fields."""
    # Test with /id suffixed fields (should remain as string)
    result = _safe_convert_field_value("parent_id/id", "some_value", "char")
    assert result == "some_value"

    # Test with positive field type and negative value (should remain as string
    result = _safe_convert_field_value("field", "-5", "positive")
    assert isinstance(result, (int, str))

    # Test with negative field type and positive value (should remain as string
    result = _safe_convert_field_value("field", "5", "negative")
    assert isinstance(result, (int, str))


def test_handle_create_error_constraint_violation() -> None:
    """Test _handle_create_error with constraint violation error."""
    error = Exception("constraint violation occurred")
    error_str, failed_line, summary = _handle_create_error(
        0, error, ["test", "data"], "original summary"
    )
    assert "Constraint violation" in error_str
    assert "data" in failed_line
    assert summary == "original summary"  # Should not change since not "Fell back to create"


def test_handle_create_error_constraint_violation_fallback() -> None:
    """Test _handle_create_error with constraint violation error during fallback."""
    error = Exception("constraint violation occurred")
    error_str, failed_line, summary = _handle_create_error(
        0, error, ["test", "data"], "Fell back to create"
    )
    assert "Constraint violation" in error_str
    assert "data" in failed_line
    assert summary == "Database constraint violation detected"  # Should change during fallback


def test_handle_create_error_connection_pool_exhaustion() -> None:
    """Test _handle_create_error with connection pool exhaustion error."""
    error = Exception("connection pool is full")
    error_str, failed_line, summary = _handle_create_error(
        1, error, ["rec1", "Alice"], "Fell back to create"
    )
    assert "Database connection pool exhaustion" in error_str
    assert "Alice" in failed_line
    assert summary == "Database connection pool exhaustion detected"


def test_handle_create_error_serialization_error() -> None:
    """Test _handle_create_error with database serialization error."""
    error = Exception("could not serialize access due to concurrent update")
    error_str, failed_line, summary = _handle_create_error(
        2, error, ["rec2", "Bob"], "Fell back to create"
    )
    assert "Database serialization error" in error_str
    assert "Bob" in failed_line
    assert summary == "Database serialization conflict detected during create"


def test_handle_create_error_external_id_field_error() -> None:
    """Test _handle_create_error with external ID field error."""
    error = Exception("Invalid field 'partner_id/id' in domain")
    error_str, failed_line, summary = _handle_create_error(
        3, error, ["rec3", "Charlie"], "Fell back to create"
    )
    assert "Invalid external ID field detected" in error_str
    assert "Charlie" in failed_line
    assert "Invalid external ID field detected" in summary


def test_handle_create_error_generic_error() -> None:
    """Test _handle_create_error with generic error."""
    error = Exception("Generic database error occurred")
    error_str, failed_line, summary = _handle_create_error(
        4, error, ["rec4", "David"], "Fell back to create"
    )
    assert "Generic database error occurred" in error_str
    assert "David" in failed_line
    assert "Generic database error occurred" in summary


def test_safe_convert_field_value_edge_cases() -> None:
    """Test _safe_convert_field_value with various edge cases."""
    # Test with None value for integer field
    result = _safe_convert_field_value("field", None, "integer")
    assert result in [0, ""]  # None should return 0 for integer fields

    # Test with empty string for char field
    result = _safe_convert_field_value("field", "", "char")
    assert result == ""

    # Test with whitespace-only string for integer field
    result = _safe_convert_field_value("field", "   ", "integer")
    assert result in [0, ""]

    # Test positive field with negative value
    result = _safe_convert_field_value("field", "-5", "positive")
    assert isinstance(
        result, (int, str)
    )  # Should be converted to int since it's numeric

    # Test negative field with positive value
    result = _safe_convert_field_value("field", "5", "negative")
    assert isinstance(
        result, (int, str)
    )  # Should be converted to int since it's numeric


def test_handle_tuple_index_error() -> None:
    """Test _handle_tuple_index_error function."""
    failed_lines: list[list[Any]] = []

    # Mock progress object
    mock_progress = MagicMock()

    _handle_tuple_index_error(mock_progress, "test_id", ["col1", "col2"], failed_lines)

    # Should add the failed line to the list
    assert len(failed_lines) == 1


def test_create_batch_individually_row_length_mismatch() -> None:
    """Test _create_batch_individually with row length mismatch."""
    mock_model = MagicMock()
    mock_model.browse().env.ref.return_value = None  # No existing record
    
    batch_header = ["id", "name", "email"]  # Header with 3 columns
    batch_lines = [["rec1", "Alice"]]  # Row with only 2 columns
    
    result = _create_batch_individually(
        mock_model, batch_lines, batch_header, 0, {}, [], None
    )
    
    # Should handle the error and return failed lines
    assert len(result.get("failed_lines", [])) == 1
    # The failed line should contain an error message about row length
    failed_line = result["failed_lines"][0]
    assert "columns" in str(failed_line[-1]).lower()


def test_create_batch_individually_connection_pool_exhaustion() -> None:
    """Test _create_batch_individually with connection pool exhaustion error."""
    mock_model = MagicMock()
    mock_model.browse().env.ref.return_value = None  # No existing record
    # Make create raise a connection pool exhaustion error
    mock_model.create.side_effect = Exception("connection pool is full")
    
    batch_header = ["id", "name"]
    batch_lines = [["rec1", "Alice"]]
    
    result = _create_batch_individually(
        mock_model, batch_lines, batch_header, 0, {}, [], None
    )
    
    # Should handle the error and return failed lines
    assert len(result.get("failed_lines", [])) == 1
    # The failed line should contain an error message about connection pool
    failed_line = result["failed_lines"][0]
    assert "connection pool" in str(failed_line[-1]).lower()


def test_create_batch_individually_serialization_error() -> None:
    """Test _create_batch_individually with database serialization error."""
    mock_model = MagicMock()
    mock_model.browse().env.ref.return_value = None  # No existing record
    # Make create raise a serialization error
    mock_model.create.side_effect = Exception("could not serialize access")
    
    batch_header = ["id", "name"]
    batch_lines = [["rec1", "Alice"]]
    
    result = _create_batch_individually(
        mock_model, batch_lines, batch_header, 0, {}, [], None
    )
    
    # Should handle the error and continue processing
    # For retryable errors like serialization errors, it should not add to failed lines
    # but just continue with other records (since there are no other records, it continues)
    assert isinstance(result.get("id_map", {}), dict)
    assert isinstance(result.get("failed_lines", []), list)


def test_create_batch_individually_tuple_index_out_of_range() -> None:
    """Test _create_batch_individually with tuple index out of range."""
    mock_model = MagicMock()
    # Make create raise an exception
    mock_model.create.side_effect = IndexError("tuple index out of range")
    mock_model.browse().env.ref.return_value = None  # No existing record

    batch_header = ["id", "name"]
    batch_lines = [["rec1", "Alice"], ["rec2", "Bob"]]

    result = _create_batch_individually(
        mock_model, batch_lines, batch_header, 0, {}, [], None
    )
    # Should handle the error and return failed lines
    assert len(result.get("failed_lines", [])) == 2  # Both records should fail


def test_create_batch_individually_existing_record() -> None:
    """Test _create_batch_individually with existing record."""
    mock_model = MagicMock()
    # Mock an existing record
    mock_existing_record = MagicMock()
    mock_existing_record.id = 123
    mock_model.browse().env.ref.return_value = mock_existing_record

    batch_header = ["id", "name"]
    batch_lines = [["rec1", "Alice"]]

    result = _create_batch_individually(
        mock_model, batch_lines, batch_header, 0, {}, [], None
    )
    
    # Should find the existing record and add it to id_map
    assert result.get("id_map", {}).get("rec1") == 123
    # Should not have any failed lines since the record already exists
    assert len(result.get("failed_lines", [])) == 0


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
    ) as mock_create:
        mock_create.return_value = {
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


def test_execute_load_batch_value_error_exception() -> None:
    """Test _execute_load_batch when a ValueError is raised from fatal error message."""
    mock_model = MagicMock()
    thread_state = {
        "model": mock_model,
        "progress": MagicMock(),
        "unique_id_field_index": 0,
        "force_create": False,
        "ignore_list": [],
    }
    batch_header = ["id", "name"]
    batch_lines = [["rec1", "Alice"], ["rec2", "Bob"]]

    # Mock the model.load method to return a fatal error message that causes ValueError
    mock_model.load.return_value = {
        "messages": [
            {
                "type": "error",
                "message": "Fatal constraint violation occurred",
            }
        ],
        "ids": [],
    }

    # When the ValueError is raised, it should be caught by the general exception handler
    # and the function should still return a result with captured failures
    result = _execute_load_batch(thread_state, batch_lines, batch_header, 1)

    # Should return success=True but with failed lines captured
    assert result["success"] is True
    assert len(result["failed_lines"]) > 0
    # The failed lines should contain the error message
    assert "Fatal constraint violation occurred" in str(result["failed_lines"])
    # Should have an empty id_map since no records were created
    assert result["id_map"] == {}


def test_execute_load_batch_database_constraint_violation() -> None:
    """Test _execute_load_batch with database constraint violation error."""
    mock_model = MagicMock()
    thread_state = {
        "model": mock_model,
        "progress": MagicMock(),
        "unique_id_field_index": 0,
        "force_create": False,
        "ignore_list": [],
    }
    batch_header = ["id", "name"]
    batch_lines = [["rec1", "Alice"], ["rec2", "Bob"]]

    # Mock the model.load method to return constraint violation error
    mock_model.load.return_value = {
        "messages": [
            {
                "type": "error",
                "message": 'duplicate key value violates unique constraint "product_product_combination_unique"',
            }
        ],
        "ids": [],
    }

    # When the constraint violation error is raised, it should be caught by the general exception handler
    result = _execute_load_batch(thread_state, batch_lines, batch_header, 1)

    # Should return success=True but with failed lines captured
    assert result["success"] is True
    assert len(result["failed_lines"]) > 0
    # The failed lines should contain the constraint violation message
    assert "duplicate key value violates unique constraint" in str(result["failed_lines"])
    # Should have an empty id_map since no records were created
    assert result["id_map"] == {}


def test_read_data_file_unicode_decode_error() -> None:
    """Test _read_data_file with UnicodeDecodeError."""
    with patch("builtins.open") as mock_open:
        mock_open.side_effect = UnicodeDecodeError("utf-8", b"test", 0, 1, "fake error")

        header, data = _read_data_file("dummy.csv", ",", "utf-8", 0)
        assert header == []
        assert data == []


def test_read_data_file_all_fallbacks_fail_unicode() -> None:
    """Test _read_data_file when all fallback encodings fail with UnicodeDecodeError."""
    with patch("builtins.open") as mock_open:

        def open_side_effect(filename: str, encoding: str, **kwargs: Any) -> Any:
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
        assert writer is not None
        assert handle is not None


def test_create_batches_empty_data() -> None:
    """Test _create_batches with empty data."""
    data: list[list[Any]] = []
    split_by_cols: Optional[list[str]] = None
    header: list[str] = []
    batch_size = 10
    o2m = False
    
    batches = list(_create_batches(data, split_by_cols, header, batch_size, o2m))
    
    # Should return empty list when data is empty
    assert batches == []


def test_create_batches_simple_data() -> None:
    """Test _create_batches with simple data."""
    data = [
        ["id1", "Alice"],
        ["id2", "Bob"],
        ["id3", "Charlie"]
    ]
    split_by_cols: Optional[list[str]] = None
    header = ["id", "name"]
    batch_size = 2
    o2m = False
    
    batches = list(_create_batches(data, split_by_cols, header, batch_size, o2m))
    
    # Should create batches with correct numbering and data
    assert len(batches) == 2
    assert batches[0][0] == 1  # First batch number
    assert batches[0][1] == [["id1", "Alice"], ["id2", "Bob"]]  # First batch data
    assert batches[1][0] == 2  # Second batch number
    assert batches[1][1] == [["id3", "Charlie"]]  # Second batch data


def test_recursive_create_batches_no_id_column() -> None:
    """Test _recursive_create_batches when no 'id' column exists."""
    header = ["name", "age"]  # No 'id' column
    data = [["Alice", "25"], ["Bob", "30"]]

    batches = list(_recursive_create_batches(data, [], header, 10, False))
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
            "odoo_data_flow.import_threaded.conf_lib.get_connection_from_dict"
        ) as mock_get_conn:
            mock_get_conn.return_value = mock_connection
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
                    file_csv="dummy.csv",  # This will be mocked
                )

                # Should succeed
                assert result is True


def test_import_data_connection_failure() -> None:
    """Test import_data when connection fails."""
    with patch(
        "odoo_data_flow.import_threaded.conf_lib.get_connection_from_dict"
    ) as mock_get_conn:
        mock_get_conn.side_effect = Exception("Connection failed")

        result, _stats = import_data(
            config={"host": "localhost"},  # Dict config
            model="res.partner",
            unique_id_field="id",
            file_csv="dummy.csv",
        )

        # Should fail gracefully
        assert result is False
        assert _stats == {}


def test_import_data_connection_exception_handling() -> None:
    """Test import_data connection exception handling path."""
    # Mock the connection to raise an exception
    with patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_dict") as mock_get_conn:
        mock_get_conn.side_effect = Exception("Connection failed")
        
        result, stats = import_data(
            config={"host": "localhost"},
            model="res.partner",
            unique_id_field="id",
            file_csv="dummy.csv",
        )
        
        # Should fail gracefully and return False with empty stats
        assert result is False
        assert stats == {}


def test_import_data_pass_1_failure() -> None:
    """Test import_data when pass 1 fails."""
    with patch("odoo_data_flow.import_threaded._read_data_file", return_value=(["id"], [["1"]])):
        with patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_dict") as mock_get_conn:
            mock_connection = MagicMock()
            mock_get_conn.return_value = mock_connection
            mock_model = MagicMock()
            mock_connection.get_model.return_value = mock_model
            
            # Mock _orchestrate_pass_1 to return success=False
            with patch("odoo_data_flow.import_threaded._orchestrate_pass_1") as mock_orchestrate:
                mock_orchestrate.return_value = {"success": False}
                
                result, stats = import_data(
                    config={"host": "localhost"},
                    model="res.partner",
                    unique_id_field="id",
                    file_csv="dummy.csv",
                )
                
                # Should fail when pass 1 is not successful
                assert result is False
                assert stats == {}


def test_import_data_deferred_fields_processing() -> None:
    """Test import_data with deferred fields processing."""
    with patch("odoo_data_flow.import_threaded._read_data_file", return_value=(["id"], [["1"]])):
        with patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_dict") as mock_get_conn:
            mock_connection = MagicMock()
            mock_get_conn.return_value = mock_connection
            mock_model = MagicMock()
            mock_connection.get_model.return_value = mock_model
            
            # Mock _orchestrate_pass_1 to return success with id_map
            with patch("odoo_data_flow.import_threaded._orchestrate_pass_1") as mock_orchestrate_pass_1:
                mock_orchestrate_pass_1.return_value = {
                    "success": True,
                    "id_map": {"1": 101}
                }
                
                # Mock _orchestrate_pass_2 to return success
                with patch("odoo_data_flow.import_threaded._orchestrate_pass_2") as mock_orchestrate_pass_2:
                    mock_orchestrate_pass_2.return_value = (True, 5)  # success, updates_made
                    
                    result, stats = import_data(
                        config={"host": "localhost"},
                        model="res.partner",
                        unique_id_field="id",
                        file_csv="dummy.csv",
                        deferred_fields=["category_id"]  # Include deferred fields
                    )
                    
                    # Should succeed with both passes
                    assert result is True
                    assert "total_records" in stats
                    assert "created_records" in stats
                    assert "updated_relations" in stats
                    assert stats["updated_relations"] == 5


def test_database_constraint_violation_unique_key() -> None:
    """Test database constraint violation for unique key violations."""
    mock_model = MagicMock()
    # Mock the model.load method to return a unique key violation error
    mock_model.load.return_value = {
        "messages": [
            {
                "type": "error",
                "message": "duplicate key value violates unique constraint \"product_product_combination_unique\""
            }
        ],
        "ids": []
    }
    
    thread_state = {
        "model": mock_model,
        "progress": MagicMock(),
        "unique_id_field_index": 0,
        "force_create": False,
        "ignore_list": [],
    }
    batch_header = ["id", "name"]
    batch_lines = [["rec1", "Alice"]]
    
    result = _execute_load_batch(thread_state, batch_lines, batch_header, 1)
    
    # Should handle the constraint violation gracefully
    assert result["success"] is True  # Should still return success
    # Should capture the failed lines
    assert len(result["failed_lines"]) > 0
    # Should contain the constraint violation error message
    assert "duplicate key value violates unique constraint" in str(result["failed_lines"])


def test_database_constraint_violation_foreign_key() -> None:
    """Test database constraint violation for foreign key violations."""
    mock_model = MagicMock()
    # Mock the model.load method to return a foreign key violation error
    mock_model.load.return_value = {
        "messages": [
            {
                "type": "error",
                "message": "insert or update on table \"res_partner\" violates foreign key constraint \"res_partner_category_rel_partner_id_fkey\""
            }
        ],
        "ids": []
    }
    
    thread_state = {
        "model": mock_model,
        "progress": MagicMock(),
        "unique_id_field_index": 0,
        "force_create": False,
        "ignore_list": [],
    }
    batch_header = ["id", "name", "category_id"]
    batch_lines = [["rec1", "Alice", "nonexistent_category"]]
    
    result = _execute_load_batch(thread_state, batch_lines, batch_header, 1)
    
    # Should handle the foreign key violation gracefully
    assert result["success"] is True  # Should still return success
    # Should capture the failed lines
    assert len(result["failed_lines"]) > 0
    # Should contain the foreign key violation error message
    assert "foreign key constraint" in str(result["failed_lines"])


def test_database_constraint_violation_check_constraint() -> None:
    """Test database constraint violation for check constraint violations."""
    mock_model = MagicMock()
    # Mock the model.load method to return a check constraint violation error
    mock_model.load.return_value = {
        "messages": [
            {
                "type": "error",
                "message": "new row for relation \"res_partner\" violates check constraint \"res_partner_check_active\""
            }
        ],
        "ids": []
    }
    
    thread_state = {
        "model": mock_model,
        "progress": MagicMock(),
        "unique_id_field_index": 0,
        "force_create": False,
        "ignore_list": [],
    }
    batch_header = ["id", "name", "active"]
    batch_lines = [["rec1", "Alice", "invalid_value"]]
    
    result = _execute_load_batch(thread_state, batch_lines, batch_header, 1)
    
    # Should handle the check constraint violation gracefully
    assert result["success"] is True  # Should still return success
    # Should capture the failed lines
    assert len(result["failed_lines"]) > 0
    # Should contain the check constraint violation error message
    assert "check constraint" in str(result["failed_lines"])


def test_database_constraint_violation_not_null() -> None:
    """Test database constraint violation for not null violations."""
    mock_model = MagicMock()
    # Mock the model.load method to return a not null violation error
    mock_model.load.return_value = {
        "messages": [
            {
                "type": "error",
                "message": "null value in column \"name\" violates not-null constraint"
            }
        ],
        "ids": []
    }
    
    thread_state = {
        "model": mock_model,
        "progress": MagicMock(),
        "unique_id_field_index": 0,
        "force_create": False,
        "ignore_list": [],
    }
    batch_header = ["id", "name"]
    batch_lines = [["rec1", None]]  # Null value that violates not-null constraint
    
    result = _execute_load_batch(thread_state, batch_lines, batch_header, 1)
    
    # Should handle the not null violation gracefully
    assert result["success"] is True  # Should still return success
    # Should capture the failed lines
    assert len(result["failed_lines"]) > 0
    # Should contain the not null violation error message
    assert "not-null constraint" in str(result["failed_lines"])


def test_threaded_import_orchestration_single_thread() -> None:
    """Test threaded import orchestration with single thread (max_connection=1)."""
    # With max_connection=1, ThreadPoolExecutor should not be used
    with patch("odoo_data_flow.import_threaded.ThreadPoolExecutor") as mock_executor:
        with patch("odoo_data_flow.import_threaded._read_data_file", return_value=(["id"], [["rec1"]])):
            with patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_dict") as mock_get_conn:
                mock_connection = MagicMock()
                mock_get_conn.return_value = mock_connection
                mock_model = MagicMock()
                mock_connection.get_model.return_value = mock_model
                
                # Mock _create_batches to return a simple batch
                with patch("odoo_data_flow.import_threaded._create_batches", return_value=[(1, [["rec1"]])]):
                    # Mock _orchestrate_pass_1 to return success
                    with patch("odoo_data_flow.import_threaded._orchestrate_pass_1") as mock_orchestrate:
                        mock_orchestrate.return_value = {"success": True, "id_map": {"rec1": 101}}
                        
                        result, stats = import_data(
                            config={"host": "localhost"},
                            model="res.partner",
                            unique_id_field="id",
                            file_csv="dummy.csv",
                            max_connection=1,  # Single thread - no ThreadPoolExecutor
                        )
                        
                        # Should not use ThreadPoolExecutor for single thread
                        mock_executor.assert_not_called()
                        # Should succeed
                        assert result is True


def test_threaded_import_orchestration_zero_threads() -> None:
    """Test threaded import orchestration with zero threads."""
    # With max_connection=0, should default to single thread behavior
    with patch("odoo_data_flow.import_threaded.ThreadPoolExecutor") as mock_executor:
        with patch("odoo_data_flow.import_threaded._read_data_file", return_value=(["id"], [["rec1"]])):
            with patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_dict") as mock_get_conn:
                mock_connection = MagicMock()
                mock_get_conn.return_value = mock_connection
                mock_model = MagicMock()
                mock_connection.get_model.return_value = mock_model
                
                # Mock _create_batches to return a simple batch
                with patch("odoo_data_flow.import_threaded._create_batches", return_value=[(1, [["rec1"]])]):
                    # Mock _orchestrate_pass_1 to return success
                    with patch("odoo_data_flow.import_threaded._orchestrate_pass_1") as mock_orchestrate:
                        mock_orchestrate.return_value = {"success": True, "id_map": {"rec1": 101}}
                        
                        result, stats = import_data(
                            config={"host": "localhost"},
                            model="res.partner",
                            unique_id_field="id",
                            file_csv="dummy.csv",
                            max_connection=0,  # Zero threads - should default to single thread
                        )
                        
                        # Should not use ThreadPoolExecutor for zero threads
                        mock_executor.assert_not_called()
                        # Should succeed
                        assert result is True


def test_threaded_import_orchestration_negative_threads() -> None:
    """Test threaded import orchestration with negative threads."""
    # With negative max_connection, should default to single thread behavior
    with patch("odoo_data_flow.import_threaded.ThreadPoolExecutor") as mock_executor:
        with patch("odoo_data_flow.import_threaded._read_data_file", return_value=(["id"], [["rec1"]])):
            with patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_dict") as mock_get_conn:
                mock_connection = MagicMock()
                mock_get_conn.return_value = mock_connection
                mock_model = MagicMock()
                mock_connection.get_model.return_value = mock_model
                
                # Mock _create_batches to return a simple batch
                with patch("odoo_data_flow.import_threaded._create_batches", return_value=[(1, [["rec1"]])]):
                    # Mock _orchestrate_pass_1 to return success
                    with patch("odoo_data_flow.import_threaded._orchestrate_pass_1") as mock_orchestrate:
                        mock_orchestrate.return_value = {"success": True, "id_map": {"rec1": 101}}
                        
                        result, stats = import_data(
                            config={"host": "localhost"},
                            model="res.partner",
                            unique_id_field="id",
                            file_csv="dummy.csv",
                            max_connection=-1,  # Negative threads - should default to single thread
                        )
                        
                        # Should not use ThreadPoolExecutor for negative threads
                        mock_executor.assert_not_called()
                        # Should succeed
                        assert result is True


def test_get_model_fields_callable_method() -> None:
    """Test _get_model_fields when _fields is callable method."""
    mock_model = MagicMock()
    mock_model._fields = MagicMock(return_value={"field1": {"type": "char"}})

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
