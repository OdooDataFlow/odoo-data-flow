"""Test the data quality checker functions."""

from unittest.mock import MagicMock, patch

from odoo_data_flow.lib import checker


class TestCheckers:
    """Test suite for the checker functions."""

    def test_id_validity_checker_success(self) -> None:
        """Tests that id_validity_checker returns True for valid data."""
        header = ["id", "name"]
        data = [["SKU-001", "A"], ["SKU-002", "B"], ["NULL", "C"]]
        check_func = checker.id_validity_checker("id", r"^SKU-\d{3}$")
        assert check_func(header, data) is True

    @patch("odoo_data_flow.lib.checker.log.warning")
    def test_id_validity_checker_failure(self, mock_log_warning: MagicMock) -> None:
        """Tests that id_validity_checker returns False for invalid data."""
        header = ["id", "name"]
        data = [["SKU-001", "A"], ["BAD-ID", "B"]]
        check_func = checker.id_validity_checker("id", r"^SKU-\d{3}$")
        assert check_func(header, data) is False
        mock_log_warning.assert_called_once()
        assert "does not match pattern" in mock_log_warning.call_args[0][0]

    @patch("odoo_data_flow.lib.checker.log.error")
    def test_id_validity_checker_bad_regex(self, mock_log_error: MagicMock) -> None:
        """Tests that id_validity_checker handles an invalid regex pattern."""
        header = ["id", "name"]
        data = [["SKU-001", "A"]]
        # This regex has an unclosed parenthesis, which is invalid
        check_func = checker.id_validity_checker("id", r"^SKU-(\d{3}$")
        assert check_func(header, data) is False
        mock_log_error.assert_called_once()
        assert "Invalid regex pattern" in mock_log_error.call_args[0][0]

    def test_id_validity_checker_with_custom_null_values(self) -> None:
        """Tests the checker with a custom list of null values."""
        header = ["id", "name"]
        # This data would fail if "N/A" were not treated as a null value
        data = [["SKU-001", "A"], ["N/A", "B"]]
        check_func = checker.id_validity_checker(
            "id", r"^SKU-\d{3}$", null_values=["N/A"]
        )
        # The check should pass because the "N/A" row is skipped
        assert check_func(header, data) is True

    def test_line_length_checker_success(self) -> None:
        """Tests that line_length_checker returns True for valid data."""
        header = ["id", "name"]
        data = [["1", "A"], ["2", "B"]]
        check_func = checker.line_length_checker(2)
        assert check_func(header, data) is True

    @patch("odoo_data_flow.lib.checker.log.warning")
    def test_line_length_checker_failure(self, mock_log_warning: MagicMock) -> None:
        """Tests that line_length_checker returns False for invalid data."""
        header = ["id", "name"]
        data = [["1", "A"], ["2", "B", "extra_col"]]  # This line is too long
        check_func = checker.line_length_checker(2)
        assert check_func(header, data) is False
        mock_log_warning.assert_called_once()
        assert "Expected 2 columns, but found 3" in mock_log_warning.call_args[0][0]

    def test_line_number_checker_success(self) -> None:
        """Tests that line_number_checker returns True for valid data."""
        header = ["id"]
        data = [["1"], ["2"]]
        check_func = checker.line_number_checker(2)
        assert check_func(header, data) is True

    @patch("odoo_data_flow.lib.checker.log.warning")
    def test_line_number_checker_failure(self, mock_log_warning: MagicMock) -> None:
        """Tests that line_number_checker returns False for invalid data."""
        header = ["id"]
        data = [["1"]]
        check_func = checker.line_number_checker(5)  # Expects 5 rows
        assert check_func(header, data) is False
        mock_log_warning.assert_called_once()
        assert "Expected 5 data rows, but found 1" in mock_log_warning.call_args[0][0]

    def test_cell_len_checker_success(self) -> None:
        """Tests that cell_len_checker returns True for valid data."""
        header = ["id", "name"]
        data = [["1", "short"], ["2", "short"]]
        check_func = checker.cell_len_checker(10)
        assert check_func(header, data) is True

    @patch("odoo_data_flow.lib.checker.log.warning")
    def test_cell_len_checker_failure(self, mock_log_warning: MagicMock) -> None:
        """Tests that cell_len_checker returns False for invalid data."""
        header = ["id", "description"]
        data = [["1", "This description is definitely way too long"]]
        check_func = checker.cell_len_checker(20)
        assert check_func(header, data) is False
        mock_log_warning.assert_called_once()
        assert "exceeds the max of 20" in mock_log_warning.call_args[0][0]

    @patch("odoo_data_flow.lib.checker.log.warning")
    def test_cell_len_checker_failure_no_header(
        self, mock_log_warning: MagicMock
    ) -> None:
        """Tests cell_len_checker failure when header is shorter than data row."""
        header = ["id"]  # Header is too short for the data row
        data = [["1", "This cell causes the error"]]
        check_func = checker.cell_len_checker(5)
        assert check_func(header, data) is False
        mock_log_warning.assert_called_once()
        # Should fall back to reporting the column number
        assert "column 2" in mock_log_warning.call_args[0][0]
