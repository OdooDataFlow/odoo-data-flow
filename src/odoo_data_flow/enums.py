"Defines preflight modes."

import enum


class PreflightMode(enum.Enum):
    """Defines the mode for running pre-flight checks."""

    NORMAL = "normal"
    """All checks are performed, interactive prompts for fixes are allowed."""
    FAIL_MODE = "fail"
    """A minimal set of critical checks are performed, no interactive prompts."""
