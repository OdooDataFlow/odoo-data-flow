"""This module provides a library of "checker" functions.

Each function is a factory that returns a new function designed to be passed
to the Processor's `.check()` method to perform data quality validations
before the transformation process begins.
"""

import re
from typing import Callable, Optional

from ..logging_config import log

# Type aliases for clarity
Header = list[str]
Data = list[list[str]]
CheckFunc = Callable[[Header, Data], bool]


def id_validity_checker(
    id_field: str, pattern: str, null_values: Optional[list[str]] = None
) -> CheckFunc:
    """ID Validity checker.

    Returns a checker that validates a specific column
    against a regex pattern.
    """
    if null_values is None:
        null_values = ["NULL"]

    def check_id_validity(header: Header, data: Data) -> bool:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            log.error(f"Invalid regex pattern provided to id_validity_checker: {e}")
            return False

        is_valid = True
        for i, line in enumerate(data, start=1):
            line_dict = dict(zip(header, line))
            id_value = line_dict.get(id_field, "")

            # Skip check if the value is considered null
            if id_value in null_values or not id_value:
                continue

            if not regex.match(id_value):
                log.warning(
                    f"Check Failed (ID Validity) on line {i}: Value "
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

    def check_line_length(header: Header, data: Data) -> bool:
        is_valid = True
        for i, line in enumerate(data, start=2):  # Start from 2 to account for header
            if len(line) != expected_length:
                log.warning(
                    f"Check Failed (Line Length) on line {i}: "
                    f"Expected {expected_length} columns, but found "
                    f"{len(line)}."
                )
                is_valid = False
        return is_valid

    return check_line_length


def line_number_checker(expected_line_count: int) -> CheckFunc:
    """Returns a checker that verifies the total number of data rows."""

    def check_line_number(header: Header, data: Data) -> bool:
        actual_line_count = len(data)
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

    def check_max_cell_len(header: Header, data: Data) -> bool:
        is_valid = True
        for i, line in enumerate(data, start=2):
            # Start from 2 to account for header
            for j, cell in enumerate(line):
                if len(cell) > max_cell_len:
                    column_name = header[j] if j < len(header) else f"column {j + 1}"
                    log.warning(
                        f"Check Failed (Cell Length) on line {i}, column "
                        f"'{column_name}': Cell length is {len(cell)}, "
                        f"which exceeds the max of {max_cell_len}."
                    )
                    is_valid = False
        return is_valid

    return check_max_cell_len
