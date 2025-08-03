"""This module contains a library of mapper functions.

Mappers are the core building blocks for data transformations. Each function
in this module is a "mapper factory" - it is a function that you call to
configure and return another function, which will then be executed by the
Processor for each row of the source data.

"""

import base64
import os
from typing import Any, Callable, Optional, Union, cast

import requests

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


def _get_field_value(line: LineDict, field: str, default: Any = None) -> Any:
    """Safely retrieves a value from the current data row.

    Args:
        line (LineDict): The current data row.
        field (str): The field name to retrieve.
        default (Any): The default value to return if the field is not found.

    Returns:
        Any: The value from the field or the default value.

    Example:
        >>> _get_field_value({'name': 'Alice', 'age': 30}, 'name')
        'Alice'
        >>> _get_field_value({'name': 'Alice', 'age': 30}, 'gender', 'unknown')
        'unknown'
    """
    value = line.get(field, default)
    log.debug(
        f"Getting field '{field}': value='{value}' from line keys: {list(line.keys())}"
    )
    return value


def _str_to_mapper(field: Any) -> MapperFunc:
    """Converts a string field name into a basic val mapper.

    If the input is not a string, it is assumed to be a valid mapper function.

    Args:
        field (Any): The field name or existing mapper.

    Returns:
        MapperFunc: A mapper function.

    Example:
        >>> _str_to_mapper('age')({'age': 25}, {})
        25
    """
    if isinstance(field, str):
        return val(field)
    return cast(MapperFunc, field)


def _list_to_mappers(args: tuple[Any, ...]) -> list[MapperFunc]:
    """Converts a list of strings or mappers into a list of mappers.

    Args:
        args (tuple): A variable number of field names or mappers.

    Returns:
        list[MapperFunc]: A list of mapper functions.

    Example:
        >>> mappers = _list_to_mappers(('name', 'age'))
        >>> [m({'name': 'Alice', 'age': 30}, {}) for m in mappers]

    """
    return [_str_to_mapper(f) for f in args]


def const(value: Any) -> MapperFunc:
    """Returns a mapper that always provides a constant value.

    Args:
        value (Any): The constant value to return.

    Returns:
        MapperFunc: A mapper function that always returns the constant value.

    Example:
        >>> const(42)({}, {})
        42
    """

    def const_fun(line: LineDict, state: StateDict) -> Any:
        return value

    return const_fun


def val(
    field: str,
    default: Any = None,
    postprocess: Callable[..., Any] = lambda x, s: x,
    skip: bool = False,
) -> MapperFunc:
    """Returns a mapper that gets a value from a specific field in the row.

    Args:
        field (str): The field name to retrieve the value from.
        default (Any): The default value if the field is not found.
        postprocess (Callable): A function to apply to the value.
        skip (bool): If True, raises SkippingError if the field value is None.

    Returns:
        MapperFunc: A mapper function that retrieves and processes the field value.

    Example:
        >>> val('age')({'age': 25}, {})
        25
        >>> val('age', default=0)({}, {})
        0
    """

    def val_fun(line: LineDict, state: StateDict) -> Any:
        value = _get_field_value(line, field)
        null_values = state.get("null_values", [])
        if str(value) in null_values:
            value = None

        if not value and skip:
            raise SkippingError(f"Missing required value for field '{field}'")

        final_value = value if value is not None else default
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
        separator (str): The string to place between each value.
        *fields (Any): A variable number of source column names or static strings.
        skip (bool): If True, raises SkippingError if the final result is empty.

    Returns:
        MapperFunc: A mapper function that returns the concatenated string.

    Example:
        >>> concat(', ', 'first_name', 'last_name')({'first_name': 'John', 'last_name': 'Doe'}, {})
        'John, Doe'
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
    """Returns a mapper that joins values but only if all values exist.

    If any of the values from the specified fields is empty, this mapper
    returns an empty string.

    Args:
        separator (str): The string to place between each value.
        *fields (Any): A variable number of source column names or static strings.

    Returns:
        MapperFunc: A mapper function that returns the concatenated string or an empty string.

    Example:
        >>> concat_mapper_all(', ', 'first_name', 'last_name')({'first_name': 'Alice', 'last_name': ''}, {})
        ''
        >>> concat_mapper_all(', ', 'first_name', 'last_name')({'first_name': 'Alice', 'last_name': 'Smith'}, {})
        'Alice, Smith'
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
        field (str): The source column to check for a truthy value.
        true_mapper (Any): The mapper to apply if the value in `field` is truthy.
        false_mapper (Any): The mapper to apply if the value in `field` is falsy.

    Returns:
        MapperFunc: A mapper function that returns the result of the chosen mapper.

    Example:
        >>> cond('is_active', const('Active'), const('Inactive'))({'is_active': True}, {})
        'Active'
        >>> cond('is_active', const('Active'), const('Inactive'))({'is_active': False}, {})
        'Inactive'
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
    3. If the value is not found in either list, the default value is returned.

    Args:
        field (str): The field name to evaluate.
        true_values (Optional[list[str]]): Values that are considered True.
        false_values (Optional[list[str]]): Values that are considered False.
        default (bool): The default boolean value to return.

    Returns:
        MapperFunc: A mapper function that returns a boolean value.

    Example:
         >>> bool_val('is_verified', ['yes', 'true'], ['no', 'false'])({'is_verified': 'yes'}, {})
         1
         >>> bool_val('is_verified', ['yes', 'true'], ['no', 'false'])({'is_verified': 'no'}, {})
         0
         >>> bool_val('is_verified', ['yes', 'true'], ['no', 'false'])({'is_verified': 'TRUE'}, {}) # case-insensitive
         0
         >>> bool_val('is_verified', ['yes', 'true'], ['no', 'false'])({'is_verified': 'maybe'}, {}) # no match
         0
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

    Example:
        >>> num('price')({'price': '1234.56'}, {})
        1234.56
        >>> num('quantity', default=0)({'quantity': ''}, {})
        0
        >>> num('age')({'age': '30'}, {})
        30
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
        col (str): The name of the column to check.

    Returns:
        MapperFunc: A mapper function that returns the column name or an empty string.

    Example:
        >>> field('product_name')({'product_name': 'Widget'}, {})
        'product_name'
        >>> field('product_name')({}, {})
        ''
    """

    def field_fun(line: LineDict, state: StateDict) -> str:
        return col if _get_field_value(line, col) else ""

    return field_fun


def m2o(prefix: str, field: str, default: str = "", skip: bool = False) -> MapperFunc:
    """Returns a mapper that creates a Many2one external ID from a field's value.

    Args:
        prefix (str): The XML ID prefix (e.g., 'my_module').
        field (str): The source column containing the value for the ID.
        default (str): The value to return if the source value is empty.
        skip (bool): If True, raises SkippingError if the source value is empty.

    Returns:
        MapperFunc: A mapper function that returns the formatted external ID.

    Example:
        >>> m2o('product', 'product_id')({'product_id': '123'}, {})
        'product.123'

        >>> m2o('product', 'product_id', 'empty')({}, {})
        'empty'

    """

    def m2o_fun(line: LineDict, state: StateDict) -> str:
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
        prefix (str): The XML ID prefix (e.g., 'my_module').
        *fields (Any): A variable number of source column names or static strings to join.
        default (str): The value to return if the final concatenated value is empty.
        skip (bool): If True, raises SkippingError if the final result is empty.

    Returns:
        MapperFunc: A mapper function that returns the formatted external ID.

    Example:
        >>> m2o_map('product', 'category', 'product_code')({'category': 'A', 'product_code': '123'}, {})
        'product.A_123'
    """
    # Assuming concat returns a callable that accepts (line: LineDict, state: StateDict)
    concat_mapper = concat("_", *fields)

    def m2o_fun(line: LineDict, state: StateDict) -> str:
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
        prefix (str): The XML ID prefix to apply to each value.
        *fields (Any): One or more source column names from which to get values.
        sep (str): The separator to use when splitting values within a single field.
        default (str): The value to return if no IDs are generated.

    Returns:
        MapperFunc: A mapper function that returns a comma-separated string of external IDs.

    Example:
        >>> m2m('tag', 'tags')({'tags': 'red, blue, green'}, {})
        'tag.red,tag.blue,tag.green'
        >>> m2m('tag', 'tags', default='tag.default')({}, {})
        'tag.default'
        >>> m2m('tag', 'tags')({'tags': ''}, {})
        ''
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
        prefix (str): The XML ID prefix to apply.
        mapper_func (MapperFunc): The inner mapper function to execute first.

    Returns:
        MapperFunc: A mapper function that returns a formatted m2m external ID list.

    Example:
        >>> def get_colors(line, state): return 'red,green,blue'
        >>> m2m_map('color', get_colors)({}, {})
        'color.red,color.green,color.blue'
    """

    def m2m_map_fun(line: LineDict, state: StateDict) -> str:
        value = mapper_func(line, state)
        return to_m2m(prefix, value)

    return m2m_map_fun


def m2o_att_name(prefix: str, att_list: list[str]) -> MapperFunc:
    """Returns a mapper that creates a dictionary of attribute-to-ID mappings.

    This is used in legacy product import workflows.

    Args:
        prefix (str): The XML ID prefix to use for the attribute IDs.
        att_list (list[str]): A list of attribute column names to check for.

    Returns:
        MapperFunc: A mapper function that returns a dictionary.

    Example:
        >>> m2o_att_name('attr', ['color', 'size'])({'color': 'red', 'size': 'M'}, {})
        {'color': 'attr.color', 'size': 'attr.size'}
        >>> m2o_att_name('attr', ['color', 'size'])({}, {})
        {}
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

    Args:
        prefix (str): The XML ID prefix to apply to each value.
        *args (Any): Field names (str) or other mapper functions.
        sep (str): The separator to use when splitting values within a single field.
        const_values (Optional[list[str]]): Optional list of constant values to include.

    Returns:
        ListMapperFunc: A mapper function that returns a list of external IDs.

    Example:
        >>> m2m_id_list('category', 'categories', const_values=['default'])({'categories': 'A,B'}, {})
        ['category.A', 'category.B', 'category.default']
        >>> m2m_id_list('color', 'colors', sep=';')({'colors': 'red;green'}, {})
        ['color.red', 'color.green']
        >>> def get_sizes(line, state): return 'S,M,L'
        >>> m2m_id_list('size', get_sizes)({}, {})
        ['size.S', 'size.M', 'size.L']
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

    Args:
        *args (Any): Field names (str) or other mapper functions.
        sep (str): The separator to use when splitting values within a single field.
        const_values (Optional[list[str]]): Optional list of constant values to include.

    Returns:
        ListMapperFunc: A mapper function that returns a list of unique values.

    Example:
        >>> m2m_value_list('colors', const_values=['red', 'blue'])({'colors': 'green,green'}, {})
        ['green', 'red', 'blue']
        >>> m2m_value_list('sizes', sep=';')({'sizes': 'S;M;L'}, {})
        ['S', 'M', 'L']
        >>> def get_categories(line, state): return 'A,B,C'
        >>> m2m_value_list(get_categories)({}, {})
        ['A', 'B', 'C']
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
        mapping_dict (dict[Any, Any]): The dictionary to use as a translation table.
        key_mapper (Any): A mapper that provides the key to look up.
        default (Any): A default value to return if the key is not found.
        m2m (bool): If True, splits the key by commas and translates each part.

    Returns:
        MapperFunc: A mapper function that returns the translated value.

    Example:
        >>> mapping = {'A': 'X', 'B': 'Y', 'C': 'Z'}
        >>> map_val(mapping, 'code')({'code': 'B'}, {})
        'Y'
        >>> map_val(mapping, 'code', default='Unknown')({'code': 'D'}, {})
        'Unknown'
        >>> mapping = {'red': 'Rouge', 'green': 'Vert', 'blue': 'Bleu'}
        >>> map_val(mapping, 'colors', m2m=True)({'colors': 'red,green'}, {})
        'Rouge,Vert'
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
        mapping (dict[str, MapperFunc]): A mapping dictionary for the related record.

    Returns:
        MapperFunc: A mapper function that returns a dictionary of the
        processed sub-record.

    Example:
        >>> def get_name(line, state): return 'Alice'
        >>> def get_age(line, state): return 30
        >>> record_mapping = {'name': get_name, 'age': get_age}
        >>> record(record_mapping)({}, {})
        {'name': 'Alice', 'age': 30}
    """

    def record_fun(line: LineDict, state: StateDict) -> dict[str, Any]:
        return {key: mapper_func(line, state) for key, mapper_func in mapping.items()}

    return record_fun


def binary(field: str, path_prefix: str = "", skip: bool = False) -> MapperFunc:
    """Returns a mapper that converts a local file to a base64 string.

    Args:
        field (str): The source column containing the path to the file.
        path_prefix (str): An optional prefix to prepend to the file path.
        skip (bool): If True, raises SkippingError if the file is not found.

    Returns:
        MapperFunc: A mapper function that returns the base64 encoded string.

    Example:
        >>> import os
        >>> # Create a dummy file for testing
        >>> with open('test_file.txt', 'w') as f:
        ...     f.write('Hello, world!')
        >>> binary('file_path')({'file_path': 'test_file.txt'}, {})[:20]
        'SGVsbG8sIHdvcmxkIQ=='
        >>> # Clean up the dummy file
        >>> os.remove('test_file.txt')
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
        field (str): The source column containing the URL.
        skip (bool): If True, raises SkippingError if the URL cannot be fetched.

    Returns:
        MapperFunc: A mapper function that returns the base64 encoded string.
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
        att_list (list[str]): A list of attribute column names to check.

    Returns:
        MapperFunc: A mapper function that returns a dictionary.

    Example:
        >>> val_att(['color', 'size'])({'color': 'red', 'size': 'M', 'other': ''}, {})
        {'color': 'red', 'size': 'M'}
        >>> val_att(['color', 'size'])({}, {})
        {}
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
        prefix (str): The XML ID prefix to use for the attribute value IDs.
        att_list (list[str]): A list of attribute column names to process.

    Returns:
        MapperFunc: A mapper function that returns a dictionary.

    Example:
        >>> m2o_att('attr', ['color', 'size'])({'color': 'red', 'size': 'M'}, {})
        {'color': 'attr.color_red', 'size': 'attr.size_M'}
        >>> m2o_att('attr', ['color', 'size'])({}, {})
        {}
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
        separator (str): The character to join the field name and value with.
        *fields (str): The attribute columns to process.

    Returns:
        MapperFunc: A mapper function that returns the concatenated string.

    Example:
        >>> concat_field_value_m2m('_', 'Color', 'Size')({'Color': 'Blue', 'Size': 'M'}, {})
        'Color_Blue,Size_M'
        >>> concat_field_value_m2m('_', 'Color', 'Size')({}, {})
        ''
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
        prefix (str): The XML ID prefix.
        *fields (str): The attribute columns to process.

    Returns:
        MapperFunc: A mapper that returns a comma-separated string of external IDs.

    Example:
        >>> m2m_attribute_value('attr', 'Color', 'Size')({'Color': 'Blue', 'Size': 'M'}, {})
        'attr.Color_Blue,attr.Size_M'
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
        prefix (str): (Unused) Kept for backward compatibility.
        *fields (Any): The attribute columns (e.g. 'Color', 'Size') to get values from.

    Returns:
        MapperFunc: A mapper that returns a comma-separated string of attribute values.

    Example:
        >>> m2m_template_attribute_value('attr', 'Color', 'Size')({'Color': 'Blue', 'Size': 'M', 'template_id': 123}, {})
        'Blue,M'
        >>> m2m_template_attribute_value('attr', 'Color', 'Size')({}, {})
        ''
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
        line_nb (int): The number of lines per chunk.

    Returns:
        Callable[[LineDict, int], int]: A function compatible with the `Processor.split` method.

    Example:
        >>> splitter = split_line_number(10)
        >>> splitter({}, 5)
        0
        >>> splitter({}, 15)
        1
    """

    def split(line: LineDict, i: int) -> int:
        return i // line_nb

    return split


def split_file_number(file_nb: int) -> Callable[[LineDict, int], int]:
    """Returns a function to split data across a fixed number of chunks.

    Args:
        file_nb (int): The total number of chunks to create.

    Returns:
        Callable[[LineDict, int], int]: A function compatible with the `Processor.split` method.

    Example:
        >>> splitter = split_file_number(3)
        >>> splitter({}, 0)
        0
        >>> splitter({}, 1)
        1
        >>> splitter({}, 2)
        2
        >>> splitter({}, 3)
        0
    """

    def split(line: LineDict, i: int) -> int:
        return i % file_nb

    return split


def path_to_image(
    field: str, path: str
) -> Callable[[dict[str, Any], dict[str, Any]], Optional[str]]:
    """Returns a mapper that converts a local file path to a base64 string.

    Args:
        field (str): The column name containing the relative path to the image.
        path (str): The base directory where the image files are located.

    Returns:
        Callable[[dict[str, Any], dict[str, Any]], Optional[str]]: A mapper function that returns the base64 encoded string or None.
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
    """Returns a mapper that downloads an image from a URL to a base64 string.

    Args:
        field (str): The source column containing the URL.

    Returns:
        Callable[[dict[str, Any], dict[str, Any]], Optional[str]]: A mapper function that returns the base64 encoded string or None.
    """

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
