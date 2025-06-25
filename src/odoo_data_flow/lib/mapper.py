"""This module contains a library of mapper functions.

Mappers are the core building blocks for data transformations. Each function
in this module is a "mapper factory" - it is a function that you call to
configure and return another function, which will then be executed by the
Processor for each row of the source data.
"""

import base64
import os
from typing import Any, Callable

import requests

from ..logging_config import log
from .internal.exceptions import SkippingError
from .internal.tools import to_m2m, to_m2o

# Type alias for clarity
LineDict = dict[str, Any]
StateDict = dict[str, Any]
MapperFunc = Callable[[LineDict, StateDict], Any]

# --- Helper Functions ---


def _get_field_value(line: LineDict, field: str, default: Any = "") -> Any:
    """Safely retrieves a value from the current data row."""
    return line.get(field, default) or default


def _str_to_mapper(field: Any) -> MapperFunc:
    """Converts a string field name into a basic val mapper."""
    if isinstance(field, str):
        return val(field)
    return field


def _list_to_mappers(args: tuple) -> list[MapperFunc]:
    """Converts a list of strings or mappers into a list of mappers."""
    return [_str_to_mapper(f) for f in args]


# --- Basic Mappers ---


def const(value: Any) -> MapperFunc:
    """Returns a mapper that always provides a constant value."""

    def const_fun(line: LineDict, state: StateDict) -> Any:
        return value

    return const_fun


def val(
    field: str,
    default: Any = "",
    postprocess: Callable = lambda x, s: x,
    skip: bool = False,
) -> MapperFunc:
    """Returns a mapper that gets a value from a specific field in the row."""

    def val_fun(line: LineDict, state: StateDict) -> Any:
        value = _get_field_value(line, field)
        if not value and skip:
            raise SkippingError(f"Missing required value for field '{field}'")

        # Pass both value and state to the postprocess function
        return postprocess(value or default, state)

    return val_fun


# --- Combining Mappers ---


def concat(separator: str, *fields: Any, skip: bool = False) -> MapperFunc:
    """Concatenate mapper.

    Returns a mapper that joins values from multiple fields or static strings.
    If `skip` is True, it will raise a SkippingError if the result is empty.
    """
    mappers = _list_to_mappers(fields)

    def concat_fun(line: LineDict, state: StateDict) -> str:
        values = [str(m(line, state)) for m in mappers]
        # Filter out empty strings before joining
        result = separator.join([v for v in values if v])
        if not result and skip:
            raise SkippingError(
                f"Concatenated value for fields {fields} is empty."
            )
        return result

    return concat_fun


# --- Conditional Mappers ---


def cond(field: str, true_mapper: Any, false_mapper: Any) -> MapperFunc:
    """Conditional mapper.

    Returns a mapper that applies one of two mappers based on the
    truthiness of a value in a given field.
    """
    true_m = _str_to_mapper(true_mapper)
    false_m = _str_to_mapper(false_mapper)

    def cond_fun(line: LineDict, state: StateDict) -> Any:
        if _get_field_value(line, field):
            return true_m(line, state)
        else:
            return false_m(line, state)

    return cond_fun


def bool_val(field: str, true_values: list[str]) -> MapperFunc:
    """Boolean Value mapper.

    Returns a mapper that checks if a field's value is in a list of true values.
    """

    def bool_val_fun(line: LineDict, state: StateDict) -> str:
        return "1" if _get_field_value(line, field) in true_values else "0"

    return bool_val_fun


# --- Numeric Mappers ---


def num(field: str, default: str = "0.0") -> MapperFunc:
    """Number mapper.

    Returns a mapper that converts a numeric string to a standard format,
    replacing commas with dots.
    """

    def num_fun(line: LineDict, state: StateDict) -> str:
        value = _get_field_value(line, field, default)
        return value.replace(",", ".")

    return num_fun


# --- Relational Mappers ---


def m2o_map(
    prefix: str, *fields: Any, default: str = "", skip: bool = False
) -> MapperFunc:
    """M20 Mapper.

    Returns a mapper for creating a Many2one external ID by concatenating
    a prefix and values from one or more fields.
    """
    concat_mapper = concat("_", *fields)

    def m2o_fun(line: LineDict, state: StateDict) -> str:
        value = concat_mapper(line, state)
        if not value and skip:
            raise SkippingError(
                f"Missing value for m2o_map with prefix '{prefix}'"
            )
        return to_m2o(prefix, value, default=default)

    return m2o_fun


def m2m(prefix: str, *fields: Any, sep: str = ",") -> MapperFunc:
    """M2M Mapper.

    Returns a mapper for creating a comma-separated list of Many2many
    external IDs.
    It can take multiple fields or a single field to be split.
    """

    def m2m_fun(line: LineDict, state: StateDict) -> str:
        all_values = []
        if len(fields) > 1:  # Mode 1: Multiple columns
            for field in fields:
                value = _get_field_value(line, field)
                if value:
                    all_values.append(to_m2o(prefix, value))
        elif len(fields) == 1:  # Mode 2: Single column with separator
            field = fields[0]
            value = _get_field_value(line, field)
            if value:
                all_values.extend(
                    to_m2o(prefix, v.strip()) for v in value.split(sep)
                )

        return ",".join(all_values)

    return m2m_fun


def m2m_map(prefix: str, mapper_func: MapperFunc) -> MapperFunc:
    """M2M_Map Many 2 Many Mapper.

    Returns a mapper that takes the result of another mapper and creates
    a Many2many external ID list from it.
    """

    def m2m_map_fun(line: LineDict, state: StateDict) -> str:
        # Get the value from the provided mapper function
        value = mapper_func(line, state)
        # Use the standard to_m2m helper to format it correctly
        return to_m2m(prefix, value)

    return m2m_map_fun


def m2m_id_list(prefix: str, *fields: Any, sep: str = ",") -> MapperFunc:
    """M2M ID List Mapper.

    Returns a mapper that creates a comma-separated list of Many2many
    external IDs from one or more fields. It concatenates values from the
    fields first, then splits them by the separator.
    """
    concat_m = concat("", *fields)

    def m2m_id_list_fun(line: LineDict, state: StateDict) -> str:
        value = concat_m(line, state)
        if not value:
            return ""
        values = [v.strip() for v in value.split(sep)]
        return ",".join(to_m2o(prefix, v) for v in values if v)

    return m2m_id_list_fun


def m2m_value_list(*fields: Any, sep: str = ",") -> MapperFunc:
    """M2M Value List Mapper.

    Returns a mapper that combines values from multiple fields and returns
    them as a Python list of strings, split by the separator.
    """
    concat_m = concat("", *fields)

    def m2m_value_list_fun(line: LineDict, state: StateDict) -> list[str]:
        value = concat_m(line, state)
        if not value:
            return []
        return [v.strip() for v in value.split(sep) if v.strip()]

    return m2m_value_list_fun


# --- Advanced Mappers ---


def map_val(
    mapping_dict: dict, key_mapper: Any, default: Any = "", m2m: bool = False
) -> MapperFunc:
    """Returns a mapper that translates a value using a provided dictionary."""
    key_m = _str_to_mapper(key_mapper)

    def map_val_fun(line: LineDict, state: StateDict) -> Any:
        key = key_m(line, state)
        if m2m and isinstance(key, str):
            keys = [k.strip() for k in key.split(",")]
            return ",".join([str(mapping_dict.get(k, default)) for k in keys])
        return mapping_dict.get(key, default)

    return map_val_fun


def record(mapping: dict) -> MapperFunc:
    """Returns a mapper that processes a sub-mapping for a related record.

    Used for creating one-to-many records.
    """

    def record_fun(line: LineDict, state: StateDict) -> dict:
        # This function returns a dictionary that the Processor will understand
        # as a related record to be created.
        return {
            key: mapper_func(line, state)
            for key, mapper_func in mapping.items()
        }

    return record_fun


# --- Binary Mappers ---


def binary(field: str, path_prefix: str = "", skip: bool = False) -> MapperFunc:
    """Binary mapper.

    Returns a mapper that reads a local file path from a field,
    and converts the file content to a base64 string.
    """

    def binary_fun(line: LineDict, state: StateDict) -> str:
        filepath = _get_field_value(line, field)
        if not filepath:
            return ""

        full_path = os.path.join(path_prefix, filepath)
        try:
            with open(full_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError as e:
            if skip:
                raise SkippingError(f"File not found at '{full_path}'") from e
            log.warning(f"File not found at '{full_path}', skipping.")
            return ""

    return binary_fun


def binary_url_map(field: str, skip: bool = False) -> MapperFunc:
    """Binary url mapper.

    Returns a mapper that reads a URL from a field, downloads the content,
    and converts it to a base64 string.
    """

    def binary_url_fun(line: LineDict, state: StateDict) -> str:
        url = _get_field_value(line, field)
        if not url:
            return ""

        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()  # Raises an exception for 4xx/5xx errors
            return base64.b64encode(res.content).decode("utf-8")
        except requests.exceptions.RequestException as e:
            if skip:
                raise SkippingError(
                    f"Cannot fetch file at URL '{url}': {e}"
                ) from e
            log.warning(f"Cannot fetch file at URL '{url}': {e}")
            return ""

    return binary_url_fun


# --- Legacy / Specialized Mappers ---


def m2m_template_attribute_value(prefix: str, *fields: Any) -> MapperFunc:
    """Legace m2m Template Attribute mapper.

    Legacy mapper for creating complex XML IDs for product attribute values.
    """
    concat_m = concat("_", *fields)

    def m2m_attribute_fun(line: LineDict, state: StateDict) -> str:
        value = concat_m(line, state)
        if not value:
            return ""
        return to_m2o(prefix, value)

    return m2m_attribute_fun


# --- Split Mappers ---


def split_line_number(line_nb: int) -> Callable:
    """Split line number.

    Returns a function for the Processor's split method that creates a new
    chunk every 'line_nb' lines.
    """

    def split(line: LineDict, i: int) -> int:
        return i // line_nb

    return split


def split_file_number(file_nb: int) -> Callable:
    """Split file number.

    Returns a function for the Processor's split method that distributes
    records across a fixed number of 'file_nb' chunks.
    """

    def split(line: LineDict, i: int) -> int:
        return i % file_nb

    return split
