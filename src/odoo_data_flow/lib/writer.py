"""Handles writing failed records to CSV files."""

import csv
from pathlib import Path
from typing import Any

from .internal.ui import _show_error_panel


def write_relational_failures_to_csv(
    model: str,
    field: str,
    original_filename: str,
    failed_records: list[dict[str, Any]],
) -> None:
    """Writes failed relational link records to a dedicated CSV file.

    Args:
        model: The main Odoo model being imported (e.g., 'res.partner').
        field: The relational field that failed (e.g., 'category_id').
        original_filename: The path to the original source CSV file.
        failed_records: A list of dictionaries, each representing a failed link.
    """
    if not failed_records:
        return

    fail_filename = f"{Path(original_filename).stem}_relations_fail.csv"
    fail_filepath = Path(original_filename).parent / fail_filename

    try:
        file_exists = fail_filepath.exists()
        with open(fail_filepath, "a", newline="", encoding="utf-8") as f:
            header = [
                "model",
                "field",
                "parent_external_id",
                "related_external_id",
                "error_reason",
            ]
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerows(failed_records)

    except OSError as e:
        _show_error_panel(
            "File Write Error", f"Could not write to fail file {fail_filepath}: {e}"
        )
