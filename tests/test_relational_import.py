"""Tests for the direct relational import strategy."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
from rich.progress import Progress

from odoo_data_flow.lib import relational_import


@patch("odoo_data_flow.lib.relational_import.cache.load_id_map")
def test_run_direct_relational_import(
    mock_load_id_map: MagicMock,
    tmp_path: Path,
) -> None:
    """Verify the direct relational import workflow."""
    # Arrange
    source_df = pl.DataFrame(
        {
            "id": ["p1", "p2"],
            "name": ["Partner 1", "Partner 2"],
            "category_id": ["cat1,cat2", "cat2,cat3"],
        }
    )
    mock_load_id_map.return_value = pl.DataFrame(
        {"external_id": ["cat1", "cat2", "cat3"], "db_id": [11, 12, 13]}
    )

    strategy_details = {
        "relation_table": "res.partner.category.rel",
        "relation_field": "partner_id",
        "relation": "category_id",
    }
    id_map = {"p1": 1, "p2": 2}
    progress = Progress()
    task_id = progress.add_task("test")

    # Act
    result = relational_import.run_direct_relational_import(
        "dummy.conf",
        "res.partner",
        "category_id",
        strategy_details,
        source_df,
        id_map,
        1,
        10,
        progress,
        task_id,
        "test.csv",
    )

    # Assert
    assert result is not None
    assert isinstance(result, dict)
    assert "file_csv" in result
    assert "model" in result
    assert "unique_id_field" in result
    assert mock_load_id_map.call_count == 1


@patch("odoo_data_flow.lib.relational_import.conf_lib.get_connection_from_config")
@patch("odoo_data_flow.lib.relational_import.cache.load_id_map")
def test_run_write_tuple_import(
    mock_load_id_map: MagicMock,
    mock_get_connection_from_config: MagicMock,
    tmp_path: Path,
) -> None:
    """Verify the write tuple import workflow."""
    # Arrange
    source_df = pl.DataFrame(
        {
            "id": ["p1", "p2"],
            "name": ["Partner 1", "Partner 2"],
            "category_id": ["cat1,cat2", "cat2,cat3"],
        }
    )
    mock_load_id_map.return_value = pl.DataFrame(
        {"external_id": ["cat1", "cat2", "cat3"], "db_id": [11, 12, 13]}
    )

    mock_connection = MagicMock()
    mock_get_connection_from_config.return_value = mock_connection
    mock_model = MagicMock()
    mock_connection.get_model.return_value = mock_model
    mock_model.export_data.return_value = {"datas": [["Test"]]}

    strategy_details = {
        "relation_table": "res.partner.category.rel",
        "relation_field": "partner_id",
        "relation": "category_id",
    }
    id_map = {"p1": 1, "p2": 2}
    progress = Progress()
    task_id = progress.add_task("test")

    # Act
    result = relational_import.run_write_tuple_import(
        "dummy.conf",
        "res.partner",
        "category_id",
        strategy_details,
        source_df,
        id_map,
        1,
        10,
        progress,
        task_id,
        "test.csv",
    )

    # Assert
    assert result is True
    assert mock_load_id_map.call_count == 1


@patch("odoo_data_flow.lib.relational_import.conf_lib.get_connection_from_config")
def test_resolve_related_ids_failure(
    mock_get_connection_from_config: MagicMock,
) -> None:
    """Test that _resolve_related_ids returns None on failure."""
    mock_connection = MagicMock()
    mock_get_connection_from_config.return_value = mock_connection
    mock_model = MagicMock()
    mock_connection.get_model.return_value = mock_model
    mock_model.search_read.side_effect = Exception("Test error")

    result = relational_import._resolve_related_ids(
        "dummy.conf", "res.partner.category", pl.Series(["cat1", "cat2"])
    )

    assert result is None


@patch("odoo_data_flow.lib.relational_import.conf_lib.get_connection_from_dict")
def test_resolve_related_ids_with_dict(mock_get_conn_dict: MagicMock) -> None:
    """Test _resolve_related_ids with a dictionary config."""
    mock_connection = MagicMock()
    mock_get_conn_dict.return_value = mock_connection
    mock_model = MagicMock()
    mock_connection.get_model.return_value = mock_model
    mock_model.search_read.return_value = [
        {"module": "base", "name": "partner_category_1", "res_id": 11},
        {"module": "base", "name": "partner_category_2", "res_id": 12},
    ]

    result = relational_import._resolve_related_ids(
        {"hostname": "localhost"},
        "res.partner.category",
        pl.Series(["cat1", "cat2"]),
    )

    assert result is not None
    # The function returns a DataFrame with external_id and db_id columns
    assert result.height == 2
    # Check that the DataFrame contains the expected data
    assert "external_id" in result.columns
    assert "db_id" in result.columns
    # Check the values in the DataFrame
    external_ids = result["external_id"].to_list()
    db_ids = result["db_id"].to_list()
    assert "partner_category_1" in external_ids
    assert "partner_category_2" in external_ids
    assert 11 in db_ids
    assert 12 in db_ids


@patch("odoo_data_flow.lib.relational_import.conf_lib.get_connection_from_config")
def test_resolve_related_ids_connection_error(
    mock_get_connection_from_config: MagicMock,
) -> None:
    """Test that _resolve_related_ids returns None on connection error."""
    mock_get_connection_from_config.side_effect = Exception("Connection error")

    result = relational_import._resolve_related_ids(
        "dummy.conf", "res.partner.category", pl.Series(["cat1", "cat2"])
    )

    assert result is None


@patch("odoo_data_flow.lib.relational_import.conf_lib.get_connection_from_config")
@patch("odoo_data_flow.lib.relational_import.cache.load_id_map")
def test_run_write_o2m_tuple_import(
    mock_load_id_map: MagicMock,
    mock_get_connection_from_config: MagicMock,
) -> None:
    """Test write O2M tuple import."""
    # Arrange
    source_df = pl.DataFrame(
        {
            "id": ["p1", "p2"],
            "name": ["Partner 1", "Partner 2"],
            "child_ids": [
                '[{"name": "Child 1"}, {"name": "Child 2"}]',
                '[{"name": "Child 3"}]',
            ],
        }
    )
    mock_load_id_map.return_value = pl.DataFrame(
        {"external_id": ["p1", "p2"], "db_id": [1, 2]}
    )

    mock_connection = MagicMock()
    mock_get_connection_from_config.return_value = mock_connection
    mock_model = MagicMock()
    mock_connection.get_model.return_value = mock_model
    mock_model.export_data.return_value = {"datas": [["Test"]]}

    strategy_details = {
        "relation": "res.partner",
    }
    id_map = {"p1": 1, "p2": 2}
    progress = Progress()
    task_id = progress.add_task("test")

    # Act
    result = relational_import.run_write_o2m_tuple_import(
        "dummy.conf",
        "res.partner",
        "child_ids",
        strategy_details,
        source_df,
        id_map,
        1,
        10,
        progress,
        task_id,
        "test.csv",
    )

    # Assert
    assert result is True


class TestQueryRelationInfoFromOdoo:
    """Tests for the _query_relation_info_from_odoo function."""

    @patch("odoo_data_flow.lib.relational_import.conf_lib.get_connection_from_config")
    def test_query_relation_info_from_odoo_success(
        self, mock_get_connection: MagicMock
    ) -> None:
        """Test successful query of relation info from Odoo."""
        # Arrange
        mock_connection = MagicMock()
        mock_get_connection.return_value = mock_connection
        mock_model = MagicMock()
        mock_connection.get_model.return_value = mock_model
        mock_model.search_read.return_value = [
            {
                "name": "product_template_attribute_line_rel",
                "model": "product.template",
            }
        ]

        # Act
        result = relational_import._query_relation_info_from_odoo(
            "dummy.conf", "product.template", "product.attribute.value"
        )

        # Assert
        assert result is not None
        assert result[0] == "product_template_attribute_line_rel"
        assert result[1] == "product_template_id"
        mock_get_connection.assert_called_once_with(config_file="dummy.conf")
        mock_model.search_read.assert_called_once()

    @patch("odoo_data_flow.lib.relational_import.conf_lib.get_connection_from_config")
    def test_query_relation_info_from_odoo_no_results(
        self, mock_get_connection: MagicMock
    ) -> None:
        """Test query of relation info from Odoo when no relations are found."""
        # Arrange
        mock_connection = MagicMock()
        mock_get_connection.return_value = mock_connection
        mock_model = MagicMock()
        mock_connection.get_model.return_value = mock_model
        mock_model.search_read.return_value = []

        # Act
        result = relational_import._query_relation_info_from_odoo(
            "dummy.conf", "product.template", "product.attribute.value"
        )

        # Assert
        assert result is None
        mock_get_connection.assert_called_once_with(config_file="dummy.conf")
        mock_model.search_read.assert_called_once()

    @patch("odoo_data_flow.lib.relational_import.conf_lib.get_connection_from_config")
    def test_query_relation_info_from_odoo_value_error_handling(
        self, mock_get_connection: MagicMock
    ) -> None:
        """Test query of relation info from Odoo with ValueError handling."""
        # Arrange
        mock_connection = MagicMock()
        mock_get_connection.return_value = mock_connection
        mock_model = MagicMock()
        mock_connection.get_model.return_value = mock_model
        # Simulate Odoo raising a ValueError with a field validation error
        # that includes ir.model.relation
        mock_model.search_read.side_effect = ValueError(
            "Invalid field 'comodel' in domain [('model', '=', 'product.template')]"
            " for model ir.model.relation"
        )

        # Act
        result = relational_import._query_relation_info_from_odoo(
            "dummy.conf", "product.template", "product.attribute.value"
        )

        # Assert
        assert result is None
        mock_get_connection.assert_called_once_with(config_file="dummy.conf")
        mock_model.search_read.assert_called_once()

    @patch("odoo_data_flow.lib.relational_import.conf_lib.get_connection_from_config")
    def test_query_relation_info_from_odoo_general_exception(
        self, mock_get_connection: MagicMock
    ) -> None:
        """Test query of relation info from Odoo with general exception handling."""
        # Arrange
        mock_get_connection.side_effect = Exception("Connection failed")

        # Act
        result = relational_import._query_relation_info_from_odoo(
            "dummy.conf", "product.template", "product.attribute.value"
        )

        # Assert
        assert result is None

    @patch("odoo_data_flow.lib.relational_import.conf_lib.get_connection_from_dict")
    def test_query_relation_info_from_odoo_with_dict_config(
        self, mock_get_connection: MagicMock
    ) -> None:
        """Test query of relation info from Odoo with dictionary configuration."""
        # Arrange
        mock_connection = MagicMock()
        mock_get_connection.return_value = mock_connection
        mock_model = MagicMock()
        mock_connection.get_model.return_value = mock_model
        mock_model.search_read.return_value = [
            {
                "name": "product_template_attribute_line_rel",
                "model": "product.template",
            }
        ]

        config_dict = {"hostname": "localhost", "database": "test_db"}

        # Act
        result = relational_import._query_relation_info_from_odoo(
            config_dict, "product.template", "product.attribute.value"
        )

        # Assert
        assert result is not None
        assert result[0] == "product_template_attribute_line_rel"
        assert result[1] == "product_template_id"
        mock_get_connection.assert_called_once_with(config_dict)
        mock_model.search_read.assert_called_once()


class TestDeriveMissingRelationInfo:
    """Tests for the _derive_missing_relation_info function."""

    def test_derive_missing_relation_info_with_all_info(self) -> None:
        """Test derive missing relation info when all info is already present."""
        # Act
        result = relational_import._derive_missing_relation_info(
            "dummy.conf",
            "product.template",
            "attribute_line_ids",
            "product_template_attribute_line_rel",
            "product_template_id",
            "product.attribute.value",
        )

        # Assert
        assert result[0] == "product_template_attribute_line_rel"
        assert result[1] == "product_template_id"

    @patch("odoo_data_flow.lib.relational_import._query_relation_info_from_odoo")
    def test_derive_missing_relation_info_without_table(
        self, mock_query: MagicMock
    ) -> None:
        """Test derive missing relation info when table is missing."""
        # Arrange
        mock_query.return_value = ("derived_table", "derived_field")

        # Act
        result = relational_import._derive_missing_relation_info(
            "dummy.conf",
            "product.template",
            "attribute_line_ids",
            None,  # Missing table
            "product_template_id",
            "product.attribute.value",
        )

        # Assert
        assert result[0] == "derived_table"
        assert result[1] == "product_template_id"
        mock_query.assert_called_once()

    @patch("odoo_data_flow.lib.relational_import._query_relation_info_from_odoo")
    def test_derive_missing_relation_info_without_field(
        self, mock_query: MagicMock
    ) -> None:
        """Test derive missing relation info when field is missing."""
        # Arrange
        mock_query.return_value = (
            "product_template_attribute_line_rel",
            "derived_field",
        )

        # Act
        result = relational_import._derive_missing_relation_info(
            "dummy.conf",
            "product.template",
            "attribute_line_ids",
            "product_template_attribute_line_rel",
            None,  # Missing field
            "product.attribute.value",
        )

        # Assert
        assert result[0] == "product_template_attribute_line_rel"
        assert result[1] == "derived_field"
        mock_query.assert_called_once()

    @patch("odoo_data_flow.lib.relational_import._query_relation_info_from_odoo")
    def test_derive_missing_relation_info_without_both(
        self, mock_query: MagicMock
    ) -> None:
        """Test derive missing relation info when both table and field are missing."""
        # Arrange
        mock_query.return_value = ("derived_table", "derived_field")

        # Act
        result = relational_import._derive_missing_relation_info(
            "dummy.conf",
            "product.template",
            "attribute_line_ids",
            None,  # Missing table
            None,  # Missing field
            "product.attribute.value",
        )

        # Assert
        assert result[0] == "derived_table"
        assert result[1] == "derived_field"
        mock_query.assert_called_once()

    @patch("odoo_data_flow.lib.relational_import._query_relation_info_from_odoo")
    def test_derive_missing_relation_info_query_returns_none(
        self, mock_query: MagicMock
    ) -> None:
        """Test derive missing relation info when query returns None."""
        # Arrange
        mock_query.return_value = None

        # Act
        result = relational_import._derive_missing_relation_info(
            "dummy.conf",
            "product.template",
            "attribute_line_ids",
            None,  # Missing table
            None,  # Missing field
            "product.attribute.value",
        )

        # Assert
        # Should fall back to derivation logic
        assert result[0] is not None
        assert result[1] is not None
        mock_query.assert_called_once()


class TestDeriveRelationInfo:
    """Tests for the _derive_relation_info function."""

    def test_derive_relation_info_known_mapping(self) -> None:
        """Test derive relation info with a known self-referencing field mapping."""
        # Act
        result = relational_import._derive_relation_info(
            "product.template", "optional_product_ids", "product.template"
        )

        # Assert
        assert result[0] == "product_optional_rel"
        assert result[1] == "product_template_id"

    def test_derive_relation_info_derived_mapping(self) -> None:
        """Test derive relation info with derived mapping."""
        # Act
        result = relational_import._derive_relation_info(
            "product.template", "attribute_line_ids", "product.attribute.value"
        )

        # Assert
        assert result[0] == "product_attribute_value_product_template_rel"
        assert result[1] == "product_template_id"

    def test_derive_relation_info_reverse_order(self) -> None:
        """Test derive relation info with reversed model order."""
        # Act
        result = relational_import._derive_relation_info(
            "product.attribute.value",  # Reversed order
            "attribute_line_ids",
            "product.template",
        )

        # Assert
        assert result[0] == "product_attribute_value_product_template_rel"
        assert result[1] == "product_attribute_value_id"
