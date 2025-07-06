"""Test the Failure Handling mechanism."""

import csv
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from odoo_data_flow.importer import run_import


def test_two_step_failure_handling(tmp_path: Path) -> None:
    """Tests the complete two-tier failure handling process.

    This test verifies that:
    1. A normal run writes the entire failed batch to a `_fail.csv` file.
    2. A `--fail` run processes the `_fail.csv` file.
    3. Records that still fail are written to a final, timestamped `_failed.csv`
       file with an added `_ERROR_REASON` column.
    """
    # --- 1. Setup: Create mock data and a mock Odoo connection ---

    source_file = tmp_path / "source_data.csv"
    model_name = "my.test.model"
    model_filename = model_name.replace(".", "_")
    intermediate_fail_file = tmp_path / f"{model_filename}_fail.csv"

    header = ["id", "name", "value"]
    # We will make the record with id='my_import.rec_02' fail on the second pass
    source_data = [
        ["my_import.rec_01", "Record 1", "100"],
        ["my_import.rec_02", "Record 2 (will fail again)", "200"],
        ["my_import.rec_03", "Record 3", "300"],
    ]

    with open(source_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(source_data)

    # This mock simulates the Odoo model's `load` method
    mock_model_load = MagicMock()

    # Define the behavior for the mock `load` method
    def load_side_effect(
        header: list[str], data: list[list[Any]], **kwargs: Any
    ) -> dict[str, Any]:
        # First pass: fail if it's a batch import
        if len(data) > 1:
            return {"messages": [{"message": "Generic batch import error"}]}
        # Second pass: succeed for some, fail for a specific record
        else:
            record_id = data[0][0]
            if record_id == "my_import.rec_02":
                return {
                    "messages": [
                        {
                            "record": 0,
                            "message": "Validation Error: "
                            "The value '200' is not valid for this field.",
                        }
                    ]
                }
            # Simulate success for other records
            return {"ids": [123]}

    mock_model_load.side_effect = load_side_effect

    # This mock simulates the odoo-client-lib connection
    mock_connection = MagicMock()
    mock_model_obj = MagicMock()
    mock_model_obj.load = mock_model_load
    mock_connection.get_model.return_value = mock_model_obj

    # --- 2. First Pass: Run the initial import ---

    with patch(
        "odoo_data_flow.import_threaded.conf_lib.get_connection_from_config",
        return_value=mock_connection,
    ):
        run_import(
            config="dummy_config.conf",
            filename=str(source_file),
            model=model_name,
            fail=False,
            separator=",",
        )

    # --- Assertions for the First Pass ---
    assert intermediate_fail_file.exists(), "Intermediate _fail.csv was not created"

    with open(intermediate_fail_file, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        header_fail1 = next(reader)
        data_fail1 = list(reader)

    assert header_fail1 == header
    assert len(data_fail1) == 3, (
        "The entire failed batch should be in the _fail.csv file"
    )
    assert data_fail1[1][1] == "Record 2 (will fail again)"  # Check content integrity

    # --- 3. Second Pass: Run the import with the --fail flag ---

    with patch(
        "odoo_data_flow.import_threaded.conf_lib.get_connection_from_config",
        return_value=mock_connection,
    ):
        run_import(
            config="dummy_config.conf",
            filename=str(
                source_file
            ),  # The original filename is still used to derive paths
            model=model_name,
            fail=True,
            separator=",",
        )

    # --- Assertions for the Second Pass ---
    # Find the final, timestamped failure file
    final_fail_files = list(tmp_path.glob("*_failed.csv"))
    assert len(final_fail_files) == 1, (
        "The final timestamped _failed.csv file was not created"
    )
    final_fail_file = final_fail_files[0]

    with open(final_fail_file, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        header_fail2 = next(reader)
        data_fail2 = list(reader)

    assert "_ERROR_REASON" in header_fail2, "The _ERROR_REASON column is missing"
    assert len(data_fail2) == 1, (
        "Only the single permanently failing record should be in the final file"
    )

    failed_record = data_fail2[0]
    error_reason_index = header_fail2.index("_ERROR_REASON")

    assert failed_record[0] == "my_import.rec_02"
    assert (
        failed_record[error_reason_index]
        == "Validation Error: The value '200' is not valid for this field."
    )
