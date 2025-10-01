"""Final coverage improvement tests for import_threaded.py."""

from typing import Any
from unittest.mock import patch

from odoo_data_flow.import_threaded import (
    _handle_create_error,
    _read_data_file,
    _safe_convert_field_value,
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
    result = _safe_convert_field_value("parent_id/id", "res_partner_1", "char")
    assert result == "res_partner_1"  # Should return string as-is for /id fields


def test_handle_create_error_tuple_index_out_of_range() -> None:
    """Test _handle_create_error with tuple index out of range."""
    error = Exception("tuple index out of range")
    error_str, failed_line, summary = _handle_create_error(
        0, error, ["test", "data"], "original summary"
    )
    assert "Tuple unpacking error" in error_str


def test_safe_convert_field_value_edge_cases() -> None:
    """Test _safe_convert_field_value with various edge cases."""
    # Test with None value for numeric field
    result = _safe_convert_field_value("field", None, "integer")
    assert result == 0  # Should return 0 for None in numeric field

    # Test with empty string for char field
    result = _safe_convert_field_value("field", "", "char")
    assert result == ""
