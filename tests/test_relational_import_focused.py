"""Focused tests for relational_import to improve coverage."""

import tempfile
from unittest.mock import Mock, patch

import polars as pl

from odoo_data_flow.lib.relational_import import (
    _derive_missing_relation_info,
    _resolve_related_ids,
)


class TestResolveRelatedIds:
    """Test _resolve_related_ids function."""

    @patch("odoo_data_flow.lib.relational_import.conf_lib")
    @patch("odoo_data_flow.lib.relational_import.cache")
    def test_resolve_related_ids_success(
        self, mock_cache: Mock, mock_conf_lib: Mock
    ) -> None:
        """Test resolving related IDs successfully."""
        # Mock cache behavior
        mock_cache.load_id_map.return_value = None  # Force fallback to bulk resolution

        # Mock connection
        mock_connection = Mock()
        mock_model = Mock()
        mock_connection.get_model.return_value = mock_model
        mock_model.search_read.return_value = [{"res_id": 1, "name": "Test"}]
        mock_conf_lib.get_connection_from_config.return_value = mock_connection

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
            f.write("[test]\nhostname=localhost\n")
            config_file = f.name

        result = _resolve_related_ids(
            config=config_file,
            related_model="res.partner",
            external_ids=pl.Series(["test_id"]),
        )
        assert result is not None

    @patch("odoo_data_flow.lib.relational_import.conf_lib")
    @patch("odoo_data_flow.lib.relational_import.cache")
    def test_resolve_related_ids_empty_result(
        self, mock_cache: Mock, mock_conf_lib: Mock
    ) -> None:
        """Test resolving related IDs when no records found."""
        mock_cache.load_id_map.return_value = None

        mock_connection = Mock()
        mock_model = Mock()
        mock_connection.get_model.return_value = mock_model
        mock_model.search_read.return_value = []
        mock_conf_lib.get_connection_from_config.return_value = mock_connection

        with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
            f.write("[test]\nhostname=localhost\n")
            config_file = f.name

        result = _resolve_related_ids(
            config=config_file,
            related_model="res.partner",
            external_ids=pl.Series(["nonexistent"]),
        )
        assert result is None

    @patch("odoo_data_flow.lib.relational_import.conf_lib")
    @patch("odoo_data_flow.lib.relational_import.cache")
    def test_resolve_related_ids_exception(
        self, mock_cache: Mock, mock_conf_lib: Mock
    ) -> None:
        """Test resolving related IDs when an exception occurs."""
        mock_cache.load_id_map.return_value = None

        mock_connection = Mock()
        mock_model = Mock()
        mock_connection.get_model.return_value = mock_model
        mock_model.search_read.side_effect = Exception("Connection error")
        mock_conf_lib.get_connection_from_config.return_value = mock_connection

        with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
            f.write("[test]\nhostname=localhost\n")
            config_file = f.name

        result = _resolve_related_ids(
            config=config_file,
            related_model="res.partner",
            external_ids=pl.Series(["test"]),
        )
        assert result is None


class TestDeriveMissingRelationInfo:
    """Test _derive_missing_relation_info function."""

    @patch("odoo_data_flow.lib.relational_import.conf_lib")
    def test_derive_missing_relation_info_success(self, mock_conf_lib: Mock) -> None:
        """Test deriving missing relation info successfully."""
        mock_connection = Mock()
        mock_model = Mock()
        mock_connection.get_model.return_value = mock_model
        mock_model.search_read.return_value = [
            {
                "relation_table": "res_partner_category_rel",
                "relation_field": "partner_id",
                "column1": "partner_id",
                "column2": "category_id",
            }
        ]
        mock_conf_lib.get_connection_from_config.return_value = mock_connection

        with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
            f.write("[test]\nhostname=localhost\n")
            config_file = f.name

        result = _derive_missing_relation_info(
            config=config_file,
            model="res.partner.category",
            field="category_id",
            relational_table="res_partner_res_partner_category_rel",
            owning_model_fk="partner_id",
            related_model_fk="category_id",
        )
        assert result is not None
        # Function returns a tuple (relational_table, owning_model_fk)
        relational_table, owning_model_fk = result
        # When both values are already provided, they should be returned as-is
        assert relational_table == "res_partner_res_partner_category_rel"
        assert owning_model_fk == "partner_id"

    @patch("odoo_data_flow.lib.relational_import.conf_lib")
    def test_derive_missing_relation_info_no_result(self, mock_conf_lib: Mock) -> None:
        """Test deriving missing relation info when no records found."""
        mock_connection = Mock()
        mock_model = Mock()
        mock_connection.get_model.return_value = mock_model
        mock_model.search_read.return_value = []
        mock_conf_lib.get_connection_from_config.return_value = mock_connection

        with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
            f.write("[test]\nhostname=localhost\n")
            config_file = f.name

        result = _derive_missing_relation_info(
            config=config_file,
            model="res.partner.category",
            field="category_id",
            relational_table="res_partner_res_partner_category_rel",
            owning_model_fk="partner_id",
            related_model_fk="category_id",
        )
        assert result is not None

    @patch("odoo_data_flow.lib.relational_import.conf_lib")
    def test_derive_missing_relation_info_exception(self, mock_conf_lib: Mock) -> None:
        """Test deriving missing relation info when an exception occurs."""
        mock_connection = Mock()
        mock_model = Mock()
        mock_connection.get_model.return_value = mock_model
        mock_model.search_read.side_effect = Exception("Database error")
        mock_conf_lib.get_connection_from_config.return_value = mock_connection

        with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
            f.write("[test]\nhostname=localhost\n")
            config_file = f.name

        result = _derive_missing_relation_info(
            config=config_file,
            model="res.partner.category",
            field="category_id",
            relational_table="res_partner_res_partner_category_rel",
            owning_model_fk="partner_id",
            related_model_fk="category_id",
        )
        assert result is not None
