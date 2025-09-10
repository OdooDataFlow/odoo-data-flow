"""This module provides sorting strategies for CSV data using Polars."""

import tempfile
from typing import Optional, Union

import polars as pl

from ..logging_config import log
from .internal.ui import _show_error_panel


def sort_for_self_referencing(
    file_path: str,
    id_column: str,
    parent_column: str,
    encoding: str = "utf-8",
    separator: str = ",",
) -> Optional[Union[str, bool]]:
    """Sorts a CSV file for self-referencing hierarchies.

    This function reads a CSV file and checks if it contains a self-referencing
    hierarchical relationship (e.g., a 'parent_id' column that refers to
    values in the 'id' column). If it does, it sorts the data to ensure
    parent records (those with a null or empty parent_column) appear before
    child records.

    The sorted data is written to a new temporary file, and the path to this
    file is returned. If no sorting is needed or possible, it returns None.
    If there was an error reading the file, it returns False.

    Args:
        file_path (str): The path to the source CSV file.
        id_column (str): The name of the unique identifier column.
        parent_column (str): The name of the column containing the parent reference.
        encoding (str): The encoding of the CSV file.
        separator (str): The field separator used in the CSV file.

    Returns:
        Optional[Union[str, bool]]: The path to the temporary sorted CSV file if sorting
        was performed, None if no sorting is needed or possible, or False if
        there was an error reading the file.
    """
    try:
        # For the sort function, we only care about being able to read the file
        # well enough to detect self-referencing hierarchies. We don't need
        # to parse all columns perfectly, so we use a very tolerant approach.
        df = pl.read_csv(
            file_path,
            separator=separator,
            encoding=encoding,
            truncate_ragged_lines=True,
            infer_schema_length=0,  # Don't infer schema, treat everything as string
        )
    except (FileNotFoundError, pl.exceptions.NoDataError) as e:
        _show_error_panel(
            "File Read Error", f"Could not read the file {file_path}: {e}"
        )
        return False  # Return False to indicate an error occurred
    except (pl.exceptions.ComputeError, pl.exceptions.ShapeError) as e:
        # For schema/validation errors, we don't want to abort the import
        # These should be handled by the field validation preflight check
        log.warning(f"Could not fully parse file {file_path} for sorting: {e}")
        return None  # Return None to indicate no sorting needed/possible

    if id_column not in df.columns or parent_column not in df.columns:
        return None

    # Ensure consistent data types for comparison
    parent_ids = df.get_column(parent_column).drop_nulls().unique().cast(pl.Utf8)
    all_ids = df.get_column(id_column).unique().cast(pl.Utf8)

    if not parent_ids.is_in(all_ids.to_list()).all():
        # Not a self-referencing hierarchy within this file
        return None

    # Sort the DataFrame: null parents first
    sorted_df = df.sort(
        pl.col(parent_column).is_null(), parent_column, descending=[True, False]
    )

    # Write to a temporary file with the same separator as the original
    temp_file = tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".csv", newline=""
    )
    sorted_df.write_csv(temp_file.name, separator=separator)
    return temp_file.name
