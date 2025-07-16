"""This module contains common, reusable functions for interacting with Odoo."""

from typing import Any

import polars as pl

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


def build_polars_schema(connection: Any, model: str) -> dict[str, type[pl.DataType]]:
    """Builds a Polars schema by querying an Odoo model's fields.

    This function connects to Odoo, retrieves field definitions, and maps
    Odoo field types to their corresponding Polars dtypes. This avoids
    Polars' type inference errors during CSV reading.

    Args:
        connection: An active connection object to Odoo.
        model: The name of the Odoo model (e.g., 'res.partner').

    Returns:
        A dictionary suitable for Polars' 'schema_overrides' argument.
    """
    log.info(f"Building Polars schema from Odoo model: '{model}'")

    # Mapping of Odoo field types to Polars dtypes.
    # We default to String for any complex or unknown types to ensure safe parsing.
    odoo_to_polars_map = {
        "boolean": pl.Boolean,
        "integer": pl.Int64,
        "float": pl.Float64,
        "decimal": pl.Float64,
        "char": pl.String,
        "text": pl.String,
        "html": pl.String,
        "selection": pl.String,
        "monetary": pl.Float64,
        "date": pl.Date,
        "datetime": pl.Datetime,
        # Relational fields are read as strings (external IDs)
        "many2one": pl.String,
        "many2many": pl.String,
        "one2many": pl.String,
        "many2one_reference": pl.String,
    }

    try:
        odoo_fields = connection.get_model(model).fields_get()
        schema_overrides: dict[str, pl.DataTypeClass] = {}
        for field_name, properties in odoo_fields.items():
            odoo_type = properties.get("type")
            # Use the mapping; fall back to pl.String if type is not in our map
            polars_type = odoo_to_polars_map.get(odoo_type, pl.String)
            schema_overrides[str(field_name)] = polars_type

        log.info("✅ Successfully built schema for Polars.")
        return schema_overrides

    except Exception as e:
        log.error(f"Could not build schema from Odoo: {e}. Returning empty schema.")
        return {}
