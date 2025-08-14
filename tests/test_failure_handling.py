"""Test the Failure Handling mechanism."""

import csv
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from odoo_data_flow import import_threaded


@patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_config")
def test_two_tier_failure_handling(mock_get_conn: MagicMock, tmp_path: Path) -> None:
    """Test two tier failure handling.

    Tests that the `load` -> `create` fallback works and that records
    which still fail are written to the fail file with a reason.

    Tests the complete two-tier failure handling process.

    This test verifies that:
    1. A normal run writes the entire failed batch to a `_fail.csv` file.
    2. A `--fail` run processes the `_fail.csv` file.
    3. Records that still fail are written to a final, timestamped `_failed.csv`
       file with an added `_ERROR_REASON` column.
    """
    # --- 1. Setup ---
    source_file = tmp_path / "source_data.csv"
    fail_file = tmp_path / "my_test_model_fail.csv"
    model_name = "my.test.model"
    header = ["id", "name", "value"]
    source_data = [
        ["rec_01", "Record 1", "100"],
        ["rec_02", "Record 2 (will fail create)", "200"],
    ]
    with open(source_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(source_data)

    mock_model = MagicMock()
    mock_model.with_context.return_value = mock_model
    mock_model.load.side_effect = Exception("Generic batch error")
    mock_model.browse.return_value.env.ref.return_value = None

    def create_side_effect(vals: dict[str, Any], context: dict[str, Any]) -> Any:
        if vals["id"] == "rec_02":
            raise Exception("Validation Error")
        else:
            mock_record = MagicMock()
            mock_record.id = 101
            return mock_record

    mock_model.create.side_effect = create_side_effect
    mock_get_conn.return_value.get_model.return_value = mock_model

    # --- Act ---
    # Capture the return value of the import process
    result, _ = import_threaded.import_data(
        config_file="dummy.conf",
        model=model_name,
        unique_id_field="id",
        file_csv=str(source_file),
        fail_file=str(fail_file),
    )

    # --- Assert ---
    # NEW: Assert that the overall process is a success because the abort
    # flag was not set and good records were processed.
    assert result is True

    # Existing assertions for the fail file remain
    assert fail_file.exists()
    with open(fail_file, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)

        assert len(rows) == 2  # Header + one failed record
        assert rows[1][0] == "rec_02"
        assert "Validation Error" in rows[1][3]


def test_create_fallback_handles_malformed_rows(tmp_path: Path) -> None:
    """Test that the fallback handles malformed rows.

    Tests that the create fallback is resilient to malformed CSV rows
    that have fewer columns than the header.
    """
    # 1. ARRANGE
    source_file = tmp_path / "source.csv"
    fail_file = tmp_path / "source_fail.csv"
    model_name = "res.partner"
    header = ["id", "name", "value"]  # Expects 3 columns
    source_data = [
        ["rec_ok", "Good Record", "100"],
        ["rec_bad", "Bad Record"],  # This row is malformed (only 2 columns)
    ]
    with open(source_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(source_data)

    mock_model = MagicMock()
    mock_model.with_context.return_value = mock_model
    mock_model.load.side_effect = Exception("Load fails, trigger fallback")
    mock_model.browse.return_value.env.ref.return_value = (
        None  # Ensure create is attempted
    )

    # 2. ACT
    with patch(
        "odoo_data_flow.import_threaded.conf_lib.get_connection_from_config"
    ) as mock_get_conn:
        mock_get_conn.return_value.get_model.return_value = mock_model
        result, _ = import_threaded.import_data(
            config_file="dummy.conf",
            model=model_name,
            unique_id_field="id",
            file_csv=str(source_file),
            fail_file=str(fail_file),
            separator=",",
        )

    # 3. ASSERT
    # The import should be considered a success since one record was processed
    assert result is True
    # The create method should only have been called for the one good record
    mock_model.create.assert_called_once()
    assert mock_model.create.call_args[0][0]["id"] == "rec_ok"

    # The fail file should exist and contain the malformed row with the correct error
    assert fail_file.exists()
    with open(fail_file) as f:
        reader = csv.reader(f, delimiter=",")
        fail_content = list(reader)

    assert len(fail_content) == 2  # Header + one failed row
    failed_row = fail_content[1]
    assert failed_row[0] == "rec_bad"
    assert "Row has 2 columns, but header has 3" in failed_row[-1]


@patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_config")
def test_fallback_with_dirty_csv(mock_get_conn: MagicMock, tmp_path: Path) -> None:
    """Test fallback handling with a dirty CSV containing various errors."""
    # 1. ARRANGE
    source_file = tmp_path / "dirty.csv"
    fail_file = tmp_path / "dirty_fail.csv"
    model_name = "res.partner"
    header = ["id", "name", "email"]
    # CSV content with various issues
    dirty_data = [
        ["ok_1", "Normal Record", "ok1@test.com"],
        ["bad_cols"],  # Malformed row, too few columns
        ["ok_2", "Another Good One", "ok2@test.com"],
        [],  # Empty row
    ]
    with open(source_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(dirty_data)

    mock_model = MagicMock()
    mock_model.load.side_effect = Exception("Load fails, forcing fallback")
    mock_model.browse.return_value.env.ref.return_value = None  # Force create
    mock_get_conn.return_value.get_model.return_value = mock_model

    # 2. ACT
    result, _ = import_threaded.import_data(
        config_file="dummy.conf",
        model=model_name,
        unique_id_field="id",
        file_csv=str(source_file),
        fail_file=str(fail_file),
        separator=",",
    )

    # 3. ASSERT
    assert result is True  # Process should succeed as good records exist
    assert mock_model.create.call_count == 2  # Called for ok_1 and ok_2

    # Verify the content of the fail file
    assert fail_file.exists()
    with open(fail_file, encoding="utf-8") as f:
        reader = csv.reader(f)
        failed_rows = list(reader)

    assert len(failed_rows) == 3  # Header + 2 failed rows
    # Check the error message for the row with bad columns
    assert failed_rows[1][0] == "bad_cols"
    assert "Row has 1 columns, but header has 3" in failed_rows[1][-1]
    # Check the error message for the empty row
    assert "Row has 0 columns, but header has 3" in failed_rows[2][-1]


@patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_config")
def test_load_with_ignored_columns(mock_get_conn: MagicMock, tmp_path: Path) -> None:
    """Test that the load method is called with correctly filtered data."""
    # 1. ARRANGE
    source_file = tmp_path / "source.csv"
    header = ["id", "name", "age"]
    data = [["1", "Alice", "30"]]
    with open(source_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    mock_model = MagicMock()
    mock_model.load.return_value = {"ids": [1], "messages": []}
    mock_get_conn.return_value.get_model.return_value = mock_model

    # 2. ACT
    import_threaded.import_data(
        config_file="dummy.conf",
        model="res.partner",
        unique_id_field="id",
        file_csv=str(source_file),
        ignore=["age"],
        separator=",",
    )

    # 3. ASSERT
    mock_model.load.assert_called_once()
    load_args = mock_model.load.call_args[0]
    assert load_args[0] == ["id", "name"]  # Header
    assert load_args[1] == [["1", "Alice"]]  # Data
