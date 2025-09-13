"""Tests for handling many2many fields with missing relation information."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl

from odoo_data_flow.enums import PreflightMode
from odoo_data_flow.lib import preflight, relational_import


@patch("odoo_data_flow.lib.preflight.pl.read_csv")
@patch("odoo_data_flow.lib.preflight.conf_lib.get_connection_from_config")
def test_handle_m2m_field_missing_relation_info(
    mock_conf_lib: MagicMock,
    mock_polars_read_csv: MagicMock,
    tmp_path: Path,
) -> None:
    """Verify that _handle_m2m_field works when relation info is missing."""
    mock_df_header = MagicMock()
    mock_df_header.columns = ["id", "name", "category_id"]

    # Setup a more robust mock for the chained Polars calls
    mock_df_data = MagicMock()
    (
        mock_df_data.lazy.return_value.select.return_value.select.return_value.sum.return_value.collect.return_value.item.return_value
    ) = 100
    mock_polars_read_csv.side_effect = [mock_df_header, mock_df_data]

    mock_model = mock_conf_lib.return_value.get_model.return_value
    mock_model.fields_get.return_value = {
        "id": {"type": "integer"},
        "name": {"type": "char"},
        "category_id": {
            "type": "many2many",
            "relation": "res.partner.category",
            # Missing relation_table and relation_field
        },
    }
    import_plan: dict[str, Any] = {}
    result = preflight.deferral_and_strategy_check(
        preflight_mode=PreflightMode.NORMAL,
        model="res.partner",
        filename="file.csv",
        config="",
        import_plan=import_plan,
    )
    assert result is True
    assert "category_id" in import_plan["deferred_fields"]
    assert import_plan["strategies"]["category_id"]["strategy"] == "write_tuple"
    # Should include relation info even when missing from Odoo metadata
    assert "relation" in import_plan["strategies"]["category_id"]
    # Should include None values for missing fields
    assert import_plan["strategies"]["category_id"]["relation_table"] is None
    assert import_plan["strategies"]["category_id"]["relation_field"] is None


@patch("odoo_data_flow.lib.relational_import.conf_lib.get_connection_from_config")
@patch("odoo_data_flow.lib.relational_import._resolve_related_ids")
def test_run_write_tuple_import_derives_missing_info(
    mock_resolve_ids: MagicMock,
    mock_get_conn: MagicMock,
) -> None:
    """Verify that run_write_tuple_import derives missing relation info."""
    # Arrange
    source_df = pl.DataFrame(
        {
            "id": ["p1", "p2"],
            "name": ["Partner 1", "Partner 2"],
            "category_id": ["cat1,cat2", "cat2,cat3"],
        }
    )
    mock_resolve_ids.return_value = pl.DataFrame(
        {"external_id": ["cat1", "cat2", "cat3"], "db_id": [11, 12, 13]}
    )
    mock_rel_model = MagicMock()
    mock_get_conn.return_value.get_model.return_value = mock_rel_model

    # Strategy details with missing relation_table and relation_field
    strategy_details = {
        "relation_table": None,  # Missing
        "relation_field": None,  # Missing
        "relation": "res.partner.category",
    }
    id_map = {"p1": 1, "p2": 2}

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
        MagicMock(),  # progress
        MagicMock(),  # task_id
        "source.csv",
    )

    # Assert
    # Should succeed because we derive the missing information
    assert result is True
    # Should have called create on the derived relation table
    mock_get_conn.return_value.get_model.assert_called()
    mock_rel_model.create.assert_called()


@patch("odoo_data_flow.lib.relational_import.conf_lib.get_connection_from_config")
@patch("odoo_data_flow.lib.relational_import._resolve_related_ids")
def test_run_direct_relational_import_derives_missing_info(
    mock_resolve_ids: MagicMock,
    mock_get_conn: MagicMock,
) -> None:
    """Verify that run_direct_relational_import derives missing relation info."""
    # Arrange
    source_df = pl.DataFrame(
        {
            "id": ["p1", "p2"],
            "name": ["Partner 1", "Partner 2"],
            "category_id": ["cat1,cat2", "cat2,cat3"],
        }
    )
    mock_resolve_ids.return_value = pl.DataFrame(
        {"external_id": ["cat1", "cat2", "cat3"], "db_id": [11, 12, 13]}
    )

    # Strategy details with missing relation_table and relation_field
    strategy_details = {
        "relation_table": None,  # Missing
        "relation_field": None,  # Missing
        "relation": "res.partner.category",
    }
    id_map = {"p1": 1, "p2": 2}

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
        MagicMock(),  # progress
        MagicMock(),  # task_id
        "source.csv",
    )

    # Assert
    # Should succeed because we derive the missing information
    assert isinstance(result, dict)
    # Should have derived the relation table name
    assert "res_partner_res_partner_category_rel" in result["model"]
    # Should have derived the relation field name
    assert "res_partner_id" in result["unique_id_field"]
