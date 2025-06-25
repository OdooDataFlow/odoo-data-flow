"""Excpention handler.

This module defines custom exceptions used throughout the library.
"""

from typing import Any


class SkippingError(Exception):
    """An exception raised to signal that the current row should be skipped.

    This is used within mappers to control the data processing flow and
    intentionally filter out certain records without causing the entire
    process to fail.
    """

    def __init__(self, message: str, *args: Any):
        """Initializes the exception with a descriptive message.

        *args:
            message: The reason why the row is being skipped.
        """
        self.message = message
        # Call the parent Exception's __init__ to ensure it behaves
        # like a standard Python exception.
        super().__init__(message, *args)
