"""This module contains a library of mapper functions.

Mappers are the core building blocks for data transformations. Each function
in this module is a "mapper factory" - it is a function that you call to
configure and return another function, which will then be executed by the
Processor for each row of the source data.

"""

import base64
import os
from typing import Any, Callable, Optional, Union, cast

import requests  # type: ignore[import-untyped]

from ..logging_config import log
from .internal.exceptions import SkippingError
from .internal.tools import to_m2m, to_m2o

__all__ = [
    "binary",
    "binary_url_map",
    "bool_val",
    "concat",
    "concat_field_value_m2m",
    "concat_mapper_all",
    "cond",
    "const",
    "field",
    "m2m",
    "m2m_attribute_value",
    "m2m_id_list",
    "m2m_map",
    "m2m_template_attribute_value",
    "m2m_value_list",
    "m2o",
    "m2o_att",
    "m2o_att_name",
    "m2o_map",
    "map_val",
    "num",
    "record",
    "split_file_number",
    "split_line_number",
    "to_m2m",
    "to_m2o",
    "val",
    "val_att",
]

# Type alias for clarity
LineDict = dict[str, Any]
StateDict = dict[str, Any]
MapperFunc = Callable[[LineDict, StateDict], Any]
ListMapperFunc = Callable[[LineDict, StateDict], list[str]]


def _get_field_value(line: LineDict, field: str, default: Any = "") -> Any:
    """Safely retrieves a value from the current data row."""
    value = line.get(field, default)
    log.debug(
        f"Getting field '{field}': value='{value}' from line keys: {list(line.keys())}"
    )
    return value


def _str_to_mapper(field: Any) -> MapperFunc:
    """Converts a string field name into a basic val mapper.

    If the input is not a string, it is assumed to be a valid mapper function.
    """
    if isinstance(field, str):
        return val(field)
    return cast(MapperFunc, field)


def _list_to_mappers(args: tuple[Any, ...]) -> list[MapperFunc]:
    """Converts a list of strings or mappers into a list of mappers."""
    return [_str_to_mapper(f) for f in args]


def const(value: Any) -> MapperFunc:
    """Returns a mapper that always provides a constant value."""

    def const_fun(line: LineDict, state: StateDict) -> Any:
        return value

    return const_fun


def val(
    field: str,
    default: Any = "",
    postprocess: Callable[..., Any] = lambda x, s: x,
    skip: bool = False,
) -> MapperFunc:
    """Returns a mapper that gets a value from a specific field in the row."""

    def val_fun(line: LineDict, state: StateDict) -> Any:
        value = _get_field_value(line, field)
        if not value and skip:
            raise SkippingError(f"Missing required value for field '{field}'")

        final_value = value or default
        try:
            # Try calling postprocess with 2 arguments first
            return postprocess(final_value, state)
        except TypeError:
            # If it fails, fall back to calling with 1 argument
            return postprocess(final_value)

    return val_fun


def concat(separator: str, *fields: Any, skip: bool = False) -> MapperFunc:
    """Returns a mapper that joins values from multiple fields or static strings.

    Args:
        separator: The string to place between each value.
        *fields: A variable number of source column names or static strings.
        skip: If True, raises SkippingError if the final result is empty.

    Returns:
        MapperFunc: A mapper function that returns the concatenated string.
    """
    mappers = _list_to_mappers(fields)

    def concat_fun(line: LineDict, state: StateDict) -> str:
        values = [str(m(line, state)) for m in mappers]
        result = separator.join([v for v in values if v])
        if not result and skip:
            raise SkippingError(f"Concatenated value for fields {fields} is empty.")
        return result

    return concat_fun


def concat_mapper_all(separator: str, *fields: Any) -> MapperFunc:
    """Returns a mapper that joins values, but only if all values exist.

    If any of the values from the specified fields is empty, this mapper
    returns an empty string.

    Args:
        separator: The string to place between each value.
        *fields: A variable number of source column names or static strings.

    Returns:
        Returns:
        A mapper function that returns the concatenated string or an empty string.
    """
    mappers = _list_to_mappers(fields)

    def concat_all_fun(line: LineDict, state: StateDict) -> str:
        values = [str(m(line, state)) for m in mappers]
        if not all(values):
            return ""
        return separator.join(values)

    return concat_all_fun


def cond(field: str, true_mapper: Any, false_mapper: Any) -> MapperFunc:
    """Returns a mapper that applies one of two mappers based on a condition.

    Args:
        field: The source column to check for a truthy value.
        true_mapper: The mapper to apply if the value in `field` is truthy.
        false_mapper: The mapper to apply if the value in `field` is falsy.

    Returns:
        MapperFunc: A mapper function that returns the result of the chosen mapper.
    """
    true_m = _str_to_mapper(true_mapper)
    false_m = _str_to_mapper(false_mapper)

    def cond_fun(line: LineDict, state: StateDict) -> Any:
        if _get_field_value(line, field):
            return true_m(line, state)
        else:
            return false_m(line, state)

    return cond_fun


def bool_val(
    field: str,
    true_values: Optional[list[str]] = None,
    false_values: Optional[list[str]] = None,
    default: bool = False,
) -> MapperFunc:
    """Returns a mapper that converts a field value to a boolean '1' or '0'.

    The logic is as follows:
    1. If `true_values` is provided, any value in that list is considered True.
    2. If `false_values` is provided, any value in that list is considered False.
    3. If the value is not in either list, the truthiness of the value itself
       is used, unless `default` is set.
    4. If no lists are provided, the truthiness of the value is used.

    Args:
        field: The source column to check.
        true_values: A list of strings that should be considered `True`.
        false_values: A list of strings that should be considered `False`.
        default: The default boolean value to return if no other condition is met.

    Returns:
        MapperFunc: A mapper function that returns "1" or "0".

    """
    true_vals = true_values or []
    false_vals = false_values or []

    def bool_val_fun(line: LineDict, state: StateDict) -> str:
        value = _get_field_value(line, field)
        if true_vals and value in true_vals:
            return "1"
        if false_vals and value in false_vals:
            return "0"
        if not true_vals and not false_vals:
            return "1" if value else str(int(default))
        return str(int(default))

    return bool_val_fun


def num(
    field: str, default: Optional[Union[int, float]] = None
) -> Callable[..., Optional[Union[int, float]]]:
    """Creates a mapper that converts a value to a native integer or float.

    This function is a factory that generates a mapper function. The returned
    mapper attempts to robustly parse a value from a source dictionary key
    into a numeric type. It handles values that are already numbers, numeric
    strings (with or without commas), or empty/null.

    Args:
        field (str): The key or column name to retrieve the value from in a
            source dictionary.
        default (Any, optional): The value to return if the source value is
            empty, null, or cannot be converted to a number. Defaults to None.

    Returns:
        Callable[..., Optional[Union[int, float]]]: A mapper function that takes a
            dictionary-like row and returns the converted numeric value (`int`
            or `float`) or the default.
    """

    def num_fun(
        line: dict[str, Any], state: dict[str, Any]
    ) -> Optional[Union[int, float]]:
        value = line.get(field)

        if value is None or value == "":
            return default

        try:
            # Convert any input to a standardized float first.
            num_val = float(str(value).replace(",", "."))

            # Return an int if it's a whole number, otherwise return the float.
            return int(num_val) if num_val.is_integer() else num_val

        except (ValueError, TypeError):
            # If any conversion fails, return the default.
            return default

    return num_fun


def field(col: str) -> MapperFunc:
    """Returns the column name itself if the column has a value.

    This is useful for some dynamic product attribute mappings.

    Args:
        col: The name of the column to check.

    Returns:
        MapperFunc: A mapper function that returns the column name or an empty string.
    """

    def field_fun(line: LineDict, state: StateDict) -> str:
        return col if _get_field_value(line, col) else ""

    return field_fun


def m2o(prefix: str, field: str, default: str = "", skip: bool = False) -> MapperFunc:
    """Returns a mapper that creates a Many2one external ID from a field's value.

    Args:
        prefix: The XML ID prefix (e.g., 'my_module').
        field: The source column containing the value for the ID.
        default: The value to return if the source value is empty.
        skip: If True, raises SkippingError if the source value is empty.

    Returns:
        A mapper function that returns the formatted external ID.
    """

    def m2o_fun(line: LineDict, state: StateDict) -> str:
        """Inner function implementing the Many2one ID mapping.

        Args:
            line: A dictionary representing the current line of data.
            state: A dictionary holding state information for the transformation
                   process.
                   This argument is part of the standard mapper function signature
                   but is not directly used by this specific mapper.

        Returns:
            The formatted external ID for the Many2one field.

        Raises:
            SkippingError: If `skip` is True and the field's value is empty.
        """
        # 'state' is included in the signature to conform to the general mapper contract
        # but is not directly used in this function's logic.
        value = _get_field_value(line, field)
        if skip and not value:
            raise SkippingError(f"Missing Value for {field}")
        return to_m2o(prefix, value, default=default)

    return m2o_fun


def m2o_map(
    prefix: str, *fields: Any, default: str = "", skip: bool = False
) -> MapperFunc:
    """Returns a mapper that creates a Many2one external ID by concatenating fields.

    This is useful when the unique identifier for a record is spread across
    multiple columns.

    Args:
        prefix: The XML ID prefix (e.g., 'my_module').
        *fields: A variable number of source column names or static strings to join.
        default: The value to return if the final concatenated value is empty.
        skip: If True, raises SkippingError if the final result is empty.

    Returns:
        A mapper function that returns the formatted external ID.
    """
    # Assuming concat returns a callable that accepts (line: LineDict, state: StateDict)
    concat_mapper = concat("_", *fields)

    def m2o_fun(line: LineDict, state: StateDict) -> str:
        """Inner function implementing the Many2one ID mapping from concatenated fields.

        Args:
            line: A dictionary representing the current line of data.
            state: A dictionary holding state information for the transformation
                   process.
                   This argument is passed to the underlying concatenation mapper.

        Returns:
            The formatted external ID for the Many2one field.

        Raises:
            SkippingError: If `skip` is True and the final concatenated value is empty.
        """
        value = concat_mapper(line, state)
        if not value and skip:
            raise SkippingError(f"Missing value for m2o_map with prefix '{prefix}'")
        return to_m2o(prefix, value, default=default)

    return m2o_fun


def m2m(prefix: str, *fields: Any, sep: str = ",", default: str = "") -> MapperFunc:
    """Returns a mapper that creates a comma-separated list of Many2many external IDs.

    It processes values from specified source columns, splitting them by 'sep'
    if they contain the separator, and applies the prefix to each resulting ID.

    Args:
        prefix: The XML ID prefix to apply to each value.
        *fields: One or more source column names from which to get values.
        sep: The separator to use when splitting values within a single field.
        default: The value to return if no IDs are generated.

    Returns:
        A mapper function that returns a comma-separated string of external IDs.
    """

    def m2m_fun(line: LineDict, state: StateDict) -> str:
        all_ids = []
        for field_name in fields:
            value = _get_field_value(line, field_name)
            if value and isinstance(value, str):
                # Always split if the value contains the separator
                # This makes behavior consistent regardless of # of fields
                current_field_ids = [
                    to_m2m(prefix, v.strip()) for v in value.split(sep) if v.strip()
                ]
                all_ids.extend(current_field_ids)

        # If no IDs are generated and default is provided, use it
        if not all_ids and default:
            return default

        return ",".join(all_ids)

    return m2m_fun


def m2m_map(prefix: str, mapper_func: MapperFunc) -> MapperFunc:
    """Returns a mapper that wraps another mapper for Many2many fields.

    It takes the comma-separated string result of another mapper and applies
    the `to_m2m` formatting to it.

    Args:
        prefix: The XML ID prefix to apply.
        mapper_func: The inner mapper function to execute first.

    Returns:
        MapperFunc: A mapper function that returns a formatted m2m external ID list.
    """

    def m2m_map_fun(line: LineDict, state: StateDict) -> str:
        value = mapper_func(line, state)
        return to_m2m(prefix, value)

    return m2m_map_fun


def m2o_att_name(prefix: str, att_list: list[str]) -> MapperFunc:
    """Returns a mapper that creates a dictionary of attribute-to-ID mappings.

    This is used in legacy product import workflows.

    Args:
        prefix: The XML ID prefix to use for the attribute IDs.
        att_list: A list of attribute column names to check for.

    Returns:
        A mapper function that returns a dictionary.
    """

    def m2o_att_fun(line: LineDict, state: StateDict) -> dict[str, str]:
        return {
            att: to_m2o(prefix, att) for att in att_list if _get_field_value(line, att)
        }

    return m2o_att_fun


def m2m_id_list(
    prefix: str,
    *args: Any,
    sep: str = ",",
    const_values: Optional[list[str]] = None,
) -> ListMapperFunc:
    """Returns a mapper for creating a list of M2M external IDs.

    This function can take either raw field names (str) or other mapper functions
    as its arguments. It processes each argument to produce an individual ID.
    If a field's value contains the separator, it will be split.
    """
    if const_values is None:
        const_values = []

    def m2m_id_list_fun(line: LineDict, state: StateDict) -> list[str]:
        all_ids: list[str] = []
        for arg in args:
            # Determine if arg is a field name or an already-created mapper
            if isinstance(arg, str):
                raw_value = _get_field_value(line, arg)
            elif callable(arg):  # Assume it's a mapper function
                try:
                    raw_value = arg(line, state)
                except (
                    TypeError
                ):  # Fallback for mappers not taking 'state' (less common now)
                    raw_value = arg(line)
            else:
                raw_value = ""  # Or raise error, depending on desired strictness

            if raw_value and isinstance(raw_value, str):
                # Always split values by separator if they contain it.
                # This ensures "Color_Black" and "Gender_Woman" are separate.
                parts = [v.strip() for v in raw_value.split(sep) if v.strip()]
                all_ids.extend([to_m2o(prefix, p) for p in parts])
            elif raw_value:  # If not string but truthy (e.g., a number from mapper.num)
                all_ids.append(to_m2o(prefix, str(raw_value)))

        # Add constant values, applying prefix
        all_ids.extend([to_m2o(prefix, cv) for cv in const_values if cv])

        # Ensure uniqueness and preserve order
        unique_ids = list(dict.fromkeys(all_ids))
        return unique_ids

    return m2m_id_list_fun


def m2m_value_list(
    *args: Any, sep: str = ",", const_values: Optional[list[str]] = None
) -> ListMapperFunc:
    """Returns a mapper that creates a Python list of unique raw values.

    It processes each argument to produce an individual raw value.
    If a field's value contains the separator, it will be split.
    """
    if const_values is None:
        const_values = []

    def m2m_value_list_fun(line: LineDict, state: StateDict) -> list[str]:
        """Returns a mapper that creates a Python list of unique values."""
        all_values: list[str] = []
        for arg in args:
            if isinstance(arg, str):
                raw_value = _get_field_value(line, arg)
            elif callable(arg):
                try:
                    raw_value = arg(line, state)
                except TypeError:
                    raw_value = arg(line)
            else:
                raw_value = ""

            if raw_value and isinstance(raw_value, str):
                parts = [v.strip() for v in raw_value.split(sep) if v.strip()]
                all_values.extend(parts)
            elif raw_value:  # If not string but truthy
                all_values.append(str(raw_value))

        all_values.extend([v.strip() for v in const_values if v.strip()])

        unique_values = list(dict.fromkeys(all_values))
        return unique_values

    return m2m_value_list_fun


def map_val(
    mapping_dict: dict[Any, Any],
    key_mapper: Any,
    default: Any = "",
    m2m: bool = False,
) -> MapperFunc:
    """Returns a mapper that translates a value using a provided dictionary.

    Args:
        mapping_dict: The dictionary to use as a translation table.
        key_mapper: A mapper that provides the key to look up.
        default: A default value to return if the key is not found.
        m2m: If True, splits the key by commas and translates each part.

    Returns:
        MapperFunc: A mapper function that returns the translated value.
    """
    key_m = _str_to_mapper(key_mapper)

    def map_val_fun(line: LineDict, state: StateDict) -> Any:
        key = key_m(line, state)
        if m2m and isinstance(key, str):
            keys = [k.strip() for k in key.split(",")]
            return ",".join([str(mapping_dict.get(k, default)) for k in keys])
        return mapping_dict.get(key, default)

    return map_val_fun


def record(mapping: dict[str, MapperFunc]) -> MapperFunc:
    """Returns a mapper that processes a sub-mapping for a related record.

    Used for creating one-to-many records (e.g., sales order lines).

    Args:
        mapping: A mapping dictionary for the related record.

    Returns:
        MapperFunc: A mapper function that returns a dictionary of the
        processed sub-record.
    """

    def record_fun(line: LineDict, state: StateDict) -> dict[str, Any]:
        return {key: mapper_func(line, state) for key, mapper_func in mapping.items()}

    return record_fun


def binary(field: str, path_prefix: str = "", skip: bool = False) -> MapperFunc:
    """Returns a mapper that converts a local file to a base64 string.

    Args:
        field: The source column containing the path to the file.
        path_prefix: An optional prefix to prepend to the file path.
        skip: If True, raises SkippingError if the file is not found.

    Returns:
        A mapper function that returns the base64 encoded string.
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
    """Returns a mapper that downloads a file from a URL and converts to base64.

    Args:
        field: The source column containing the URL.
        skip: If True, raises SkippingError if the URL cannot be fetched.

    Returns:
        A mapper function that returns the base64 encoded string.
    """

    def binary_url_fun(line: LineDict, state: StateDict) -> str:
        url = _get_field_value(line, field)
        if not url:
            return ""

        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            return base64.b64encode(res.content).decode("utf-8")
        except requests.exceptions.RequestException as e:
            if skip:
                raise SkippingError(f"Cannot fetch file at URL '{url}': {e}") from e
            log.warning(f"Cannot fetch file at URL '{url}': {e}")
            return ""

    return binary_url_fun


def val_att(att_list: list[str]) -> MapperFunc:
    """(Legacy V9-V12) Returns a dictionary of attributes that have a value.

    This is a helper for legacy product attribute workflows.

    Args:
        att_list: A list of attribute column names to check.

    Returns:
        A mapper function that returns a dictionary.
    """

    def val_att_fun(line: LineDict, state: StateDict) -> dict[str, Any]:
        return {
            att: _get_field_value(line, att)
            for att in att_list
            if _get_field_value(line, att)
        }

    return val_att_fun


def m2o_att(prefix: str, att_list: list[str]) -> MapperFunc:
    """(Legacy V9-V12) Returns a dictionary of attribute-to-ID mappings.

    This is a helper for legacy product attribute workflows where IDs for
    attribute values were manually constructed.

    Args:
        prefix: The XML ID prefix to use for the attribute value IDs.
        att_list: A list of attribute column names to process.

    Returns:
        A mapper function that returns a dictionary.
    """

    def m2o_att_fun(line: LineDict, state: StateDict) -> dict[str, str]:
        result = {}
        for att in att_list:
            value = _get_field_value(line, att)
            if value:
                id_value = f"{att}_{value}"
                result[att] = to_m2o(prefix, id_value)
        return result

    return m2o_att_fun


def concat_field_value_m2m(separator: str, *fields: str) -> MapperFunc:
    """(Legacy V9-V12) Specialized concat for attribute value IDs.

    Joins each field name with its value (e.g., 'Color' + 'Blue' -> 'Color_Blue'),
    then joins all resulting parts with a comma. This was used to create
    unique external IDs for `product.attribute.value` records.

    Args:
        separator: The character to join the field name and value with.
        *fields: The attribute columns to process.

    Returns:
        MapperFunc: A mapper function that returns the concatenated string.
    """

    def concat_fun(line: LineDict, state: StateDict) -> str:
        parts = []
        for field in fields:
            value = _get_field_value(line, field)
            if value:
                parts.append(f"{field}{separator}{value}")
        return ",".join(parts)

    return concat_fun


def m2m_attribute_value(prefix: str, *fields: str) -> MapperFunc:
    """(Legacy V9-V12) Creates a list of external IDs for attribute values.

    This is a composite mapper for the legacy product attribute workflow.

    Args:
        prefix: The XML ID prefix.
        *fields: The attribute columns to process.

    Returns:
        MapperFunc: A mapper that returns a comma-separated string of external IDs.
    """
    return m2m_map(prefix, concat_field_value_m2m("_", *fields))


def m2m_template_attribute_value(prefix: str, *fields: Any) -> MapperFunc:
    """(Modern V13+) Creates a comma-separated list of attribute values.

    This mapper concatenates the *values* of the given fields. This is used for
    the modern product attribute system where Odoo automatically
    creates the `product.attribute.value` records from the raw value names.

    It will return an empty string if the `template_id` is missing from the
    source line, preventing the creation of orphaned attribute lines.

    Args:
        prefix: (Unused) Kept for backward compatibility.
        *fields: The attribute columns (e.g. 'Color', 'Size') to get values from.

    Returns:
        MapperFunc: A mapper that returns a comma-separated string of attribute values.
    """
    concat_m = concat(",", *fields)

    def m2m_attribute_fun(line: LineDict, state: StateDict) -> str:
        # This check is critical for the modern workflow.
        if not line.get("template_id"):
            return ""
        return cast(str, concat_m(line, state))

    return m2m_attribute_fun


def split_line_number(line_nb: int) -> Callable[[LineDict, int], int]:
    """Returns a function to split data into chunks of a specific line count.

    Args:
        line_nb: The number of lines per chunk.

    Returns:
        A function compatible with the `Processor.split` method.
    """

    def split(line: LineDict, i: int) -> int:
        return i // line_nb

    return split


def split_file_number(file_nb: int) -> Callable[[LineDict, int], int]:
    """Returns a function to split data across a fixed number of chunks.

    Args:
        file_nb: The total number of chunks to create.

    Returns:
        A function compatible with the `Processor.split` method.
    """

    def split(line: LineDict, i: int) -> int:
        return i % file_nb

    return split


def path_to_image(
    field: str, path: str
) -> Callable[[dict[str, Any], dict[str, Any]], Optional[str]]:
    """Returns a mapper that converts a local file path to a base64 string.

    Args:
        field: The column name containing the relative path to the image.
        path: The base directory where the image files are located.
    """

    def _mapper(row: dict[str, Any], state: dict[str, Any]) -> Optional[str]:
        relative_path = row.get(field)
        if not relative_path:
            return None

        full_path = os.path.join(path, relative_path)
        if not os.path.exists(full_path):
            log.warning(f"Image file not found at: {full_path}")
            return None

        try:
            with open(full_path, "rb") as image_file:
                content = image_file.read()
            return base64.b64encode(content).decode("utf-8")
        except OSError as e:
            log.error(f"Could not read file {full_path}: {e}")
            return None

    return _mapper


def url_to_image(
    field: str,
) -> Callable[[dict[str, Any], dict[str, Any]], Optional[str]]:
    """Returns a mapper that downloads an image from a URL to a base64 string."""

    def _mapper(row: dict[str, Any], state: dict[str, Any]) -> Optional[str]:
        url = row.get(field)
        if not url:
            return None

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            # Raises an exception for bad status codes (4xx or 5xx)
            content = response.content
            return base64.b64encode(content).decode("utf-8")
        except requests.exceptions.RequestException as e:
            log.warning(f"Failed to download image from {url}: {e}")
            return None

    return _mapper
