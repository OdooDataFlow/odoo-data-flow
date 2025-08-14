"""Tests for the strategy dispatcher."""

from unittest.mock import MagicMock

import polars as pl
import pytest

from odoo_data_flow.lib import preflight


@pytest.fixture
def mock_odoo_adapter():
    """Fixture for a mocked OdooAdapter."""
    adapter = MagicMock()
    adapter.get_fields_metadata.return_value = {
        "name": {"type": "char"},
        "parent_id": {"type": "many2one", "relation": "res.partner"},
        "category_id": {
            "type": "many2many",
            "relation": "res.partner.category",
            "relation_table": "res_partner_res_partner_category_rel",
            "relation_field": "partner_id",
        },
        "child_ids": {"type": "one2many", "relation": "res.partner"},
    }
    return adapter


def test_strategy_dispatching(mock_odoo_adapter, tmp_path):
    """Test that the strategy dispatcher assigns the correct strategies."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "name": ["test1", "test2"],
            "parent_id/id": ["parent1", "parent2"],
            "category_id/id": ["cat1,cat2", "cat2"],
            "child_ids/id": ["child1", "child2"],
        }
    )
    filename = tmp_path / "test.csv"
    df.write_csv(filename)

    import_plan = {"deferred_fields": [], "strategies": {}}
    preflight._plan_deferrals_and_strategies(
        header=df.columns,
        odoo_fields=mock_odoo_adapter.get_fields_metadata(),
        model="res.partner",
        filename=str(filename),
        separator=",",
        import_plan=import_plan,
    )

    assert "parent_id" in import_plan["deferred_fields"]
    assert "category_id" in import_plan["deferred_fields"]
    assert "child_ids" in import_plan["deferred_fields"]

    # m2o self is handled by sorting, not a specific strategy here
    assert "parent_id" not in import_plan["strategies"]

    assert "category_id" in import_plan["strategies"]
    assert import_plan["strategies"]["category_id"]["strategy"] == "write_tuple"

    assert "child_ids" in import_plan["strategies"]
    assert import_plan["strategies"]["child_ids"]["strategy"] == "write_o2m_tuple"
