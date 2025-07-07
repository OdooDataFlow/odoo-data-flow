"""This module provides a library of "checker" functions.

Each function is a factory that returns a new function designed to be passed
to the Processor's `.check()` method to perform data quality validations
before the transformation process begins.
"""

import re
from typing import Callable, Optional

import polars as pl

from ..logging_config import log


def id_validity_checker(
    id_field: str, pattern: str, null_values: Optional[list[str]] = None
) -> Callable[[pl.DataFrame], bool]:
    """ID Validity checker.

    Returns a checker that validates a specific column
    against a regex pattern.
    """
    if null_values is None:
        null_values = ["NULL"]

    def check_id_validity(df: pl.DataFrame) -> bool:
        try:
            re.compile(pattern)
        except re.error as e:
            log.error(f"Invalid regex pattern provided to id_validity_checker: {e}")
            return False

        # Filter out null values
        non_null_df = df.filter(~pl.col(id_field).is_in(null_values))

        # Find invalid rows
        invalid_rows = non_null_df.filter(~pl.col(id_field).str.contains(pattern))

        if not invalid_rows.is_empty():
            for row in invalid_rows.iter_rows(named=True):
                log.warning(
                    f"Check Failed (ID Validity): Value "
                    f"'{row[id_field]}' in column '{id_field}' "
                    f"does not match pattern '{pattern}'."
                )
            return False
        return True

    return check_id_validity


def line_length_checker(expected_length: int) -> Callable[[pl.DataFrame], bool]:
    """Line Length Checker.

    Returns a checker that verifies each row has an exact number of columns.
    """

    def check_line_length(df: pl.DataFrame) -> bool:
        actual_length = df.width
        if actual_length != expected_length:
            log.warning(
                f"Check Failed (Line Length): "
                f"Expected {expected_length} columns, but found "
                f"{actual_length}."
            )
            return False
        return True

    return check_line_length


def line_number_checker(expected_line_count: int) -> Callable[[pl.DataFrame], bool]:
    """Returns a checker that verifies the total number of data rows."""

    def check_line_number(df: pl.DataFrame) -> bool:
        actual_line_count = len(df)
        if actual_line_count != expected_line_count:
            log.warning(
                f"Check Failed (Line Count): Expected {expected_line_count} "
                f"data rows, but found {actual_line_count}."
            )
            return False
        return True

    return check_line_number


def cell_len_checker(max_cell_len: int) -> Callable[[pl.DataFrame], bool]:
    """Cell Length Checker.

    Returns a checker that verifies no cell exceeds a maximum character length.
    """

    def check_max_cell_len(df: pl.DataFrame) -> bool:
        is_valid = True
        for col_name in df.columns:
            if df[col_name].dtype == pl.String:
                invalid_cells = df.filter(
                    pl.col(col_name).str.len_chars() > max_cell_len
                )
                if not invalid_cells.is_empty():
                    is_valid = False
                    for row in invalid_cells.iter_rows(named=True):
                        log.warning(
                            f"Check Failed (Cell Length) in column "
                            f"'{col_name}': Cell length is {len(row[col_name])}, "
                            f"which exceeds the max of {max_cell_len}."
                        )
        return is_valid

    return check_max_cell_len
