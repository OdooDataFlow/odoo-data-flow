"""This tests the common, reusable functions for interacting with Odoo."""

from unittest.mock import MagicMock, patch

from odoo_data_flow.lib.odoo_lib import get_odoo_version


def test_get_odoo_version_success() -> None:
    """Tests successful detection of an Odoo version."""
    # 1. Setup
    mock_connection = MagicMock()
    mock_ir_module = MagicMock()
    mock_ir_module.search_read.return_value = [{"latest_version": "17.0.1.0.0"}]
    mock_connection.get_model.return_value = mock_ir_module

    # 2. Action
    version = get_odoo_version(mock_connection)

    # 3. Assert
    assert version == 17
    mock_connection.get_model.assert_called_once_with("ir.module.module")


@patch("odoo_data_flow.lib.odoo_lib.log.warning")
def test_get_odoo_version_failure_on_exception(
    mock_log_warning: MagicMock,
) -> None:
    """Tests fallback behavior when version detection raises an exception."""
    # 1. Setup
    mock_connection = MagicMock()
    mock_connection.get_model.side_effect = Exception("Connection Error")

    # 2. Action
    version = get_odoo_version(mock_connection)

    # 3. Assert
    assert version == 14  # Should return the fallback value
    mock_log_warning.assert_called_once()
    assert "Could not detect Odoo version" in mock_log_warning.call_args[0][0]
