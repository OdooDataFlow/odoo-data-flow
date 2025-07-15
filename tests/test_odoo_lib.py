"""This tests the common, reusable functions for interacting with Odoo."""

from unittest.mock import MagicMock, patch

import polars as pl

from odoo_data_flow.lib.odoo_lib import build_polars_schema, get_odoo_version


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


class TestBuildPolarsSchema:
    """Groups tests for the build_polars_schema function."""

    def test_build_polars_schema_success(self) -> None:
        """Tests that Odoo fields are correctly mapped to Polars dtypes."""
        # 1. Setup
        mock_connection = MagicMock()
        mock_model_obj = MagicMock()

        # Define a fake response from Odoo's fields_get()
        fake_odoo_fields = {
            "name": {"type": "char"},
            "active": {"type": "boolean"},
            "sequence": {"type": "integer"},
            "some_float": {"type": "float"},
            "description": {"type": "text"},
            "partner_id": {"type": "many2one"},
            "some_unmapped_type": {"type": "binary"},  # This should default to String
        }
        mock_model_obj.fields_get.return_value = fake_odoo_fields
        mock_connection.get_model.return_value = mock_model_obj

        # 2. Action
        schema = build_polars_schema(mock_connection, "res.partner")

        # 3. Assert
        expected_schema = {
            "name": pl.String,
            "active": pl.Boolean,
            "sequence": pl.Int64,
            "some_float": pl.Float64,
            "description": pl.String,
            "partner_id": pl.String,
            "some_unmapped_type": pl.String,  # Verifies the fallback logic
        }
        assert schema == expected_schema
        mock_connection.get_model.assert_called_once_with("res.partner")
        mock_model_obj.fields_get.assert_called_once()

    @patch("odoo_data_flow.lib.odoo_lib.log.error")
    def test_build_polars_schema_failure(self, mock_log_error: MagicMock) -> None:
        """Tests that an empty dict is returned if fields_get fails."""
        # 1. Setup
        mock_connection = MagicMock()
        mock_model_obj = MagicMock()
        # Simulate an Odoo RPC error
        mock_model_obj.fields_get.side_effect = Exception("Odoo RPC Error")
        mock_connection.get_model.return_value = mock_model_obj

        # 2. Action
        schema = build_polars_schema(mock_connection, "res.partner")

        # 3. Assert
        assert schema == {}  # Should return an empty dict on failure
        mock_log_error.assert_called_once()
        assert "Could not build schema from Odoo" in mock_log_error.call_args[0][0]
