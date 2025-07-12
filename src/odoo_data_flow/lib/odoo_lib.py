"""This module contains common, reusable functions for interacting with Odoo."""

from typing import Any

from ..logging_config import log


def get_odoo_version(connection: Any) -> int:
    """Detects the major Odoo version from the server.

    This is useful for running version-specific logic.

    Args:
        connection: An active connection object to Odoo.

    Returns:
        The integer of the major Odoo version (e.g., 16, 17).
        Returns 14 as a fallback for legacy mode if detection fails.
    """
    try:
        ir_module = connection.get_model("ir.module.module")
        # The 'base' module version accurately reflects the core Odoo version.
        base_module_data = ir_module.search_read(
            [("name", "=", "base")], ["latest_version"], limit=1
        )
        if not base_module_data:
            raise Exception("Could not find the 'base' module.")

        # Version is usually formatted like "17.0.1.0.0"
        version_string = base_module_data[0]["latest_version"]
        major_version = int(version_string.split(".")[0])
        log.info(f"✅ Detected Odoo version: {major_version}")
        return major_version
    except Exception as e:
        log.warning(
            f"Could not detect Odoo version: {e}. Defaulting to legacy mode (<15)."
        )
        return 14
