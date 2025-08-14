"""Tests for the direct relational import strategy."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
from polars.testing import assert_frame_equal
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
        "source.csv",
    )

    # Assert
    assert isinstance(result, dict)
    assert "file_csv" in result
    assert "model" in result
    assert "unique_id_field" in result
    assert result["model"] == "res.partner.category.rel"
    assert result["unique_id_field"] == "partner_id"

    # Verify the content of the temporary CSV and cleanup
    temp_csv_path = result["file_csv"]
    try:
        df = pl.read_csv(temp_csv_path, truncate_ragged_lines=True)
        expected_df = pl.DataFrame(
            {
                "partner_id": [1, 1, 2, 2],
                "category_id/id": [11, 12, 12, 13],
            }
        )
        assert_frame_equal(df, expected_df, check_row_order=False)
    finally:
        Path(temp_csv_path).unlink(missing_ok=True)
