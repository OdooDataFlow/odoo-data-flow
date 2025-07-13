"""Test the data quality checker functions."""

from unittest.mock import MagicMock, patch

import polars as pl

from odoo_data_flow.lib import checker


class TestCheckers:
    """Test suite for the checker functions."""

    def test_id_validity_checker_success(self) -> None:
        """Tests that id_validity_checker returns True for valid data."""
        df = pl.DataFrame(
            {"id": ["SKU-001", "SKU-002", "NULL"], "name": ["A", "B", "C"]}
        )
        check_func = checker.id_validity_checker("id", r"^SKU-\d{3}$")
        assert check_func(df) is True

    @patch("odoo_data_flow.lib.checker.log.warning")
    def test_id_validity_checker_failure(self, mock_log_warning: MagicMock) -> None:
        """Tests that id_validity_checker returns False for invalid data."""
        df = pl.DataFrame({"id": ["SKU-001", "BAD-ID"], "name": ["A", "B"]})
        check_func = checker.id_validity_checker("id", r"^SKU-\d{3}$")
        assert check_func(df) is False
        mock_log_warning.assert_called_once()
        assert "does not match pattern" in mock_log_warning.call_args[0][0]

    @patch("odoo_data_flow.lib.checker.log.error")
    def test_id_validity_checker_bad_regex(self, mock_log_error: MagicMock) -> None:
        """Tests that id_validity_checker handles an invalid regex pattern."""
        df = pl.DataFrame({"id": ["SKU-001"], "name": ["A"]})
        # This regex has an unclosed parenthesis, which is invalid
        check_func = checker.id_validity_checker("id", r"^SKU-(\d{3}$")
        assert check_func(df) is False
        mock_log_error.assert_called_once()
        assert "Invalid regex pattern" in mock_log_error.call_args[0][0]

    def test_id_validity_checker_with_custom_null_values(self) -> None:
        """Tests the checker with a custom list of null values."""
        df = pl.DataFrame({"id": ["SKU-001", "N/A"], "name": ["A", "B"]})
        check_func = checker.id_validity_checker(
            "id", r"^SKU-\d{3}$", null_values=["N/A"]
        )
        # The check should pass because the "N/A" row is skipped
        assert check_func(df) is True

    def test_line_length_checker_success(self) -> None:
        """Tests that line_length_checker returns True for valid data."""
        df = pl.DataFrame({"id": ["1", "2"], "name": ["A", "B"]})
        check_func = checker.line_length_checker(2)
        assert check_func(df) is True

    @patch("odoo_data_flow.lib.checker.log.warning")
    def test_line_length_checker_failure(self, mock_log_warning: MagicMock) -> None:
        """Tests that line_length_checker returns False for invalid data."""
        df = pl.DataFrame({"id": ["1"], "name": ["A"], "extra_col": ["C"]})
        check_func = checker.line_length_checker(2)
        assert check_func(df) is False
        mock_log_warning.assert_called_once()
        assert "Expected 2 columns, but found 3" in mock_log_warning.call_args[0][0]

    def test_line_number_checker_success(self) -> None:
        """Tests that line_number_checker returns True for valid data."""
        df = pl.DataFrame({"id": ["1", "2"]})
        check_func = checker.line_number_checker(2)
        assert check_func(df) is True

    @patch("odoo_data_flow.lib.checker.log.warning")
    def test_line_number_checker_failure(self, mock_log_warning: MagicMock) -> None:
        """Tests that line_number_checker returns False for invalid data."""
        df = pl.DataFrame({"id": ["1"]})
        check_func = checker.line_number_checker(5)  # Expects 5 rows
        assert check_func(df) is False
        mock_log_warning.assert_called_once()
        assert "Expected 5 data rows, but found 1" in mock_log_warning.call_args[0][0]

    def test_cell_len_checker_success(self) -> None:
        """Tests that cell_len_checker returns True for valid data."""
        df = pl.DataFrame({"id": ["1", "2"], "name": ["short", "short"]})
        check_func = checker.cell_len_checker(10)
        assert check_func(df) is True

    @patch("odoo_data_flow.lib.checker.log.warning")
    def test_cell_len_checker_failure(self, mock_log_warning: MagicMock) -> None:
        """Tests that cell_len_checker returns False for invalid data."""
        df = pl.DataFrame(
            {
                "id": ["1"],
                "description": ["This description is definitely way too long"],
            }
        )
        check_func = checker.cell_len_checker(20)
        assert check_func(df) is False
        mock_log_warning.assert_called_once()
        assert "exceeds max length of 20" in mock_log_warning.call_args[0][0]

    @patch("odoo_data_flow.lib.checker.log.warning")
    def test_cell_len_checker_failure_no_header(
        self, mock_log_warning: MagicMock
    ) -> None:
        """Tests cell_len_checker failure when header is shorter than data row."""
        df = pl.DataFrame({"id": ["1"], "description": ["This cell causes the error"]})
        check_func = checker.cell_len_checker(5)
        assert check_func(df) is False
        mock_log_warning.assert_called_once()
        # Should fall back to reporting the column name
        assert "description" in mock_log_warning.call_args[0][0]
