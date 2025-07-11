"""This module provides a library of "checker" functions.

Each function is a factory that returns a new function designed to be passed
to the Processor's `.check()` method to perform data quality validations
before the transformation process begins.
"""

import re
from typing import Callable, Optional

import polars as pl

from ..logging_config import log

# Type aliases for clarity
CheckFunc = Callable[[pl.DataFrame], bool]


def id_validity_checker(
    id_field: str, pattern: str, null_values: Optional[list[str]] = None
) -> CheckFunc:
    """ID Validity checker.

    Returns a checker that validates a specific column
    against a regex pattern.
    """
    if null_values is None:
        null_values = ["NULL"]

    def check_id_validity(df: pl.DataFrame) -> bool:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            log.error(f"Invalid regex pattern provided to id_validity_checker: {e}")
            return False

        is_valid = True
        if id_field not in df.columns:
            log.error(f"ID field '{id_field}' not found in DataFrame.")
            return False

        for i, id_value in enumerate(df[id_field]):
            # Skip check if the value is considered null
            if id_value in null_values or not id_value:
                continue

            if not regex.match(str(id_value)):
                log.warning(
                    f"Check Failed (ID Validity) on line {i + 1}: Value "
                    f"'{id_value}' in column '{id_field}' "
                    f"does not match pattern '{pattern}'."
                )
                is_valid = False
        return is_valid

    return check_id_validity


def line_length_checker(expected_length: int) -> CheckFunc:
    """Line Length Checker.

    Returns a checker that verifies each row has an exact number of columns.
    """

    def check_line_length(df: pl.DataFrame) -> bool:
        if df.width == expected_length:
            return True
        else:
            log.warning(
                f"Check Failed (Line Length): Expected {expected_length} columns, "
                f"but found {df.width}."
            )
            return False

    return check_line_length


def line_number_checker(expected_line_count: int) -> CheckFunc:
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


def cell_len_checker(max_cell_len: int) -> CheckFunc:
    """Cell Length Checker.

    Returns a checker that verifies no cell exceeds a maximum character length.
    """

    def check_max_cell_len(df: pl.DataFrame) -> bool:
        is_valid = True
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                mask = df[col].str.len_chars() > max_cell_len
                if mask.any():
                    is_valid = False
                    offending_rows = df.filter(mask)
                    for row in offending_rows.iter_rows(named=True):
                        log.warning(
                            f"Check Failed (Cell Length) in column '{col}': "
                            f"Cell value '{row[col]}' exceeds max length of {max_cell_len}."
                        )
        return is_valid

    return check_max_cell_len
