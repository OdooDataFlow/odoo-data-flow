"""Internal dooo-flow Tools.

This module provides low-level utility functions for data formatting
and iteration,
primarily used by the mapper and processor modules.
"""

from collections.abc import Iterable
from itertools import islice
from typing import Any


def batch(iterable: Iterable[Any], size: int) -> Iterable[list[Any]]:
    """Splits an iterable into batches of a specified size.

    Args:
        iterable: The iterable to process.
        size: The desired size of each batch.

    Yields:
        A list containing the next batch of items.
    """
    source_iterator = iter(iterable)
    while True:
        batch_iterator = islice(source_iterator, size)
        # Get the first item to check if the iterator is exhausted
        try:
            first_item = next(batch_iterator)
        except StopIteration:
            return

        # Chain the first item back with the rest of the batch iterator
        # and yield the complete batch as a list.
        yield [first_item, *list(batch_iterator)]


# --- Data Formatting Tools ---


def to_xmlid(name: str) -> str:
    """Create valid xmlid.

    Sanitizes a string to make it a valid XML ID, replacing special
    characters with underscores.
    """
    if not isinstance(name, str):
        name = str(name)

    # A mapping of characters to replace.
    replacements = {".": "_", ",": "_", "\n": "_", "|": "_", " ": "_"}
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.strip()


def to_m2o(prefix: str, value: Any, default: str = "") -> str:
    """Creates a full external ID for a Many2one relationship.

    Creates a full external ID for a Many2one relationship by combining
    a prefix and a sanitized value.

    Args:
        prefix: The XML ID prefix (e.g., 'my_module').
        value: The value to be sanitized and appended to the prefix.
        default: The value to return if the input value is empty.

    Return:
        The formatted external ID (e.g., 'my_module.sanitized_value').
    """
    if not value:
        return default

    # Ensure the prefix ends with a dot,
    #  but don't add one if it's already there.
    if not prefix.endswith("."):
        prefix += "."

    return f"{prefix}{to_xmlid(value)}"


def to_m2m(prefix: str, value: str) -> str:
    """Creates a comma-separated list of external IDs .

    Creates a comma-separated list of external IDs for a Many2many relationship.
    It takes a string of comma-separated values, sanitizes each one, and
    prepends the prefix.

    Args:
        prefix: The XML ID prefix to apply to each value.
        value: A single string containing one or more values,
        separated by commas.

    Return:
        A comma-separated string of formatted external IDs.
    """
    if not value:
        return ""

    ids = [to_m2o(prefix, val.strip()) for val in value.split(",") if val.strip()]
    return ",".join(ids)
