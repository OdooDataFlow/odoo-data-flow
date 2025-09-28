"""Handles relational import strategies like m2m and o2m."""

import json
import tempfile
from typing import Any, Optional, Union

import polars as pl
from rich.progress import Progress, TaskID

from ..logging_config import log
from . import cache, conf_lib, writer


def _resolve_related_ids(  # noqa: C901
    config: Union[str, dict[str, Any]], related_model: str, external_ids: pl.Series
) -> Optional[pl.DataFrame]:
    """Resolve related ids.

    Resolves external IDs for a related model, trying cache first,
    then falling back to a bulk XML-ID resolution.
    """
    # 1. Try to load from cache
    if isinstance(config, str):
        related_model_cache = cache.load_id_map(config, related_model)
        if related_model_cache is not None:
            log.info(f"Cache hit for related model '{related_model}'.")
            return related_model_cache

    # 2. Fallback to bulk XML-ID resolution
    log.warning(
        f"Cache miss for related model '{related_model}'. "
        f"Falling back to slow XML-ID resolution."
    )
    if isinstance(config, dict):
        connection = conf_lib.get_connection_from_dict(config)
    else:
        connection = conf_lib.get_connection_from_config(config_file=config)

    id_list = external_ids.drop_nulls().unique().to_list()
    log.info(f"Resolving {len(id_list)} unique IDs for '{related_model}'...")

    # Separate database IDs from XML IDs
    db_ids = []
    xml_ids = []
    invalid_ids = []

    for id_val in id_list:
        if isinstance(id_val, str) and id_val.isdigit():
            # It's a numeric database ID
            db_ids.append(int(id_val))
        elif isinstance(id_val, str) and len(id_val) > 0:
            # It's a non-empty string that's not purely numeric - treat as XML ID
            xml_ids.append(id_val)
        else:
            # Empty or None values
            invalid_ids.append(id_val)

    if invalid_ids:
        log.warning(
            f"Skipping {len(invalid_ids)} invalid IDs for model "
            f"'{related_model}' (empty or None values). "
            f"Sample invalid IDs: {invalid_ids[:5]}"
        )
        if not db_ids and not xml_ids:
            return None

    resolved_map = {}

    # Handle database IDs directly
    if db_ids:
        log.info(f"Using {len(db_ids)} database IDs directly without XML resolution")
        # For database IDs, the "external ID" is the same as the database ID (as string)
        for db_id in db_ids:
            resolved_map[str(db_id)] = db_id

    # Handle XML IDs through traditional resolution
    if xml_ids:
        log.info(f"Resolving {len(xml_ids)} XML IDs through traditional lookup")

        # For XML IDs, we need to look them up by name
        try:
            data_model = connection.get_model("ir.model.data")

            # Build domain for XML ID lookup by name
            # We'll look for records where the name matches any of our XML IDs
            domain: list[tuple[str, str, Union[str, list[str]]]]
            if len(xml_ids) == 1:
                domain = [("name", "=", xml_ids[0])]
            else:
                domain = [("name", "in", xml_ids)]

            resolved_data = data_model.search_read(domain, ["module", "name", "res_id"])
            if not resolved_data:
                log.error(
                    f"XML-ID resolution failed for all IDs in model '{related_model}'. "
                    "This is often caused by referencing records that don't exist "
                    "or don't have external IDs assigned."
                )
                if not db_ids:
                    return None
            else:
                xml_resolved_map = {rec["name"]: rec["res_id"] for rec in resolved_data}
                resolved_map.update(xml_resolved_map)
                log.info(
                    f"Successfully resolved {len(xml_resolved_map)} XML IDs for "
                    f"model '{related_model}'."
                )
        except Exception as e:
            log.error(f"An error occurred during bulk XML-ID resolution: {e}")
            if not db_ids:
                return None

    if resolved_map:
        log.info(
            f"Successfully resolved {len(resolved_map)} IDs for model '{related_model}'"
            f" ({len(db_ids)} database IDs, {len(xml_ids)} XML IDs)."
        )
        return pl.DataFrame(
            {
                "external_id": list(resolved_map.keys()),
                "db_id": list(resolved_map.values()),
            }
        )
    return None


def _derive_missing_relation_info(
    config: Union[str, dict[str, Any]],
    model: str,
    field: str,
    relational_table: Optional[str],
    owning_model_fk: Optional[str],
    related_model_fk: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Derive missing relation table and field names if possible.

    First tries to query Odoo's ir.model.relation table to get actual relationship info.
    If that fails, falls back to derivation logic based on naming conventions.

    Args:
        config: Configuration for connecting to Odoo
        model: The owning model name
        field: The field name
        relational_table: Current relation table name (may be None)
        owning_model_fk: Current owning model foreign key field name (may be None)
        related_model_fk: Related model name (needed for derivation)

    Returns:
        Tuple of (relational_table, owning_model_fk) with derived values
        where missing, or original values if already present
    """
    # Try to derive missing information if possible
    if (not relational_table or not owning_model_fk) and related_model_fk:
        # First, try to get relation info from Odoo's ir.model.relation table
        odoo_relation_info = _query_relation_info_from_odoo(
            config, model, related_model_fk
        )

        if odoo_relation_info:
            derived_table, derived_field = odoo_relation_info
        else:
            # Fallback to the derivation logic
            derived_table, derived_field = _derive_relation_info(
                model, field, related_model_fk
            )

        # Only use derived values if we were missing them
        if not relational_table:
            log.info(f"Deriving relation_table for field '{field}': {derived_table}")
            relational_table = derived_table
        if not owning_model_fk:
            log.info(f"Deriving relation_field for field '{field}': {derived_field}")
            owning_model_fk = derived_field

    return relational_table, owning_model_fk


def _query_relation_info_from_odoo(
    config: Union[str, dict[str, Any]], model: str, related_model_fk: str
) -> Optional[tuple[str, str]]:
    """Query Odoo's ir.model.relation table to get actual relationship information.

    Args:
        config: Configuration for connecting to Odoo
        model: The owning model name
        related_model_fk: The related model name

    Returns:
        A tuple of (relation_table, relation_field) or None if not found
    """
    try:
        # Get connection to Odoo
        if isinstance(config, dict):
            connection = conf_lib.get_connection_from_dict(config)
        else:
            connection = conf_lib.get_connection_from_config(config_file=config)

        # Query ir.model.relation table
        # Look for relations where our models are involved
        relation_model = connection.get_model("ir.model.relation")

        # Search for relations involving both models
        # We need to check both orders since the relation could be defined either way
        domain = [
            "|",
            "&",
            ("model", "=", model),
            ("comodel", "=", related_model_fk),
            "&",
            ("model", "=", related_model_fk),
            ("comodel", "=", model),
        ]

        relations = relation_model.search_read(domain, ["name", "model", "comodel"])

        if relations:
            # Found matching relations, use the first one
            relation = relations[0]
            relation_table = relation["name"]

            # Determine the owning model field name based on which model is "model"
            # The owning model's foreign key in the relation table is derived
            # from its own model name, e.g., 'res.partner' -> 'res_partner_id'.
            relation_field = f"{model.replace('.', '_')}_id"

            log.info(
                f"Found relation info from ir.model.relation: "
                f"table='{relation_table}', field='{relation_field}'"
            )
            return relation_table, relation_field
        else:
            log.debug(
                f"No relation found in ir.model.relation for models "
                f"'{model}' and '{related_model_fk}'"
            )
            return None

    except Exception as e:
        log.warning(
            f"Failed to query ir.model.relation for models '{model}' and "
            f"'{related_model_fk}'. Error: {e}"
        )
        return None


def _derive_relation_info(
    model: str, field: str, related_model_fk: str
) -> tuple[str, str]:
    """Derive relation table and field names based on Odoo conventions.

    Args:
        model: The owning model name
        field: The field name
        related_model_fk: The related model name

    Returns:
        A tuple of (relation_table, relation_field)
    """
    # Derive relation table name (typically follows pattern: model1_model2_rel)
    # with models sorted alphabetically for canonical naming
    models = sorted([model.replace(".", "_"), related_model_fk.replace(".", "_")])
    derived_table = f"{models[0]}_{models[1]}_rel"

    # Derive the owning model field name (typically model_name_id)
    # In Odoo's many2many tables, column names typically use the full model name
    # with dots replaced by underscores, e.g., res.partner -> res_partner_id
    derived_field = f"{model.replace('.', '_')}_id"

    log.debug(
        f"Derived relation table: '{derived_table}' for models "
        f"'{model}' and '{related_model_fk}'"
    )

    return derived_table, derived_field


def run_direct_relational_import(
    config: Union[str, dict[str, Any]],
    model: str,
    field: str,
    strategy_details: dict[str, Any],
    source_df: pl.DataFrame,
    id_map: dict[str, int],
    worker: int,
    batch_size: int,
    progress: Progress,
    task_id: TaskID,
    original_filename: str,
) -> Optional[dict[str, Any]]:
    """Orchestrates the high-speed direct relational import."""
    progress.update(
        task_id,
        description=f"Pass 2/2: Updating relations for [bold]{field}[/bold]",
    )
    log.info(f"Running 'Direct Relational Import' for field '{field}'...")

    # Check if required keys exist
    relational_table = strategy_details.get("relation_table")
    owning_model_fk = strategy_details.get("relation_field")
    related_model_fk = strategy_details.get("relation")

    # Try to derive missing information if possible
    relational_table, owning_model_fk = _derive_missing_relation_info(
        config, model, field, relational_table, owning_model_fk, related_model_fk
    )

    # If we don't have the required information, we can't proceed with this strategy
    if not relational_table or not owning_model_fk:
        log.error(
            f"Cannot run direct relational import for field '{field}': "
            f"Missing relation_table or relation_field in strategy details."
        )
        return None

    # 1. Prepare the owning model's IDs
    owning_df = pl.DataFrame({"external_id": id_map.keys(), "db_id": id_map.values()})

    # Debug: Print available columns and the field we're looking for
    log.debug(f"Available columns in source_df: {source_df.columns}")
    log.debug(f"Looking for field: {field}")

    # Determine the actual column name to look for
    # For many2many fields, the column name in the DataFrame typically has /id suffix
    actual_field_name = field
    if f"{field}/id" in source_df.columns:
        actual_field_name = f"{field}/id"
        log.debug(f"Found external ID column: {actual_field_name}")

    # Check if the field exists in the DataFrame
    if actual_field_name not in source_df.columns:
        log.error(
            f"Field '{actual_field_name}' not found in source DataFrame. "
            f"Available columns: {source_df.columns}"
        )
        return None

    # 2. Prepare the related model's IDs using the resolver
    all_related_ext_ids = (
        source_df.get_column(actual_field_name).str.split(",").explode()
    )
    if related_model_fk is None:
        log.error(
            f"Cannot resolve related IDs: Missing relation in strategy details "
            f"for field '{field}'."
        )
        return None
    related_model_df = _resolve_related_ids(
        config, related_model_fk, all_related_ext_ids
    )
    if related_model_df is None:
        log.error(f"Could not resolve IDs for related model '{related_model_fk}'.")
        return None

    # 3. Create the link table DataFrame
    link_df = source_df.select(["id", actual_field_name]).rename({"id": "external_id"})
    link_df = link_df.with_columns(pl.col(actual_field_name).str.split(",")).explode(
        actual_field_name
    )

    # Join to get DB IDs for the owning model
    link_df = link_df.join(owning_df, on="external_id", how="inner").rename(
        {"db_id": owning_model_fk}
    )

    # Join to get DB IDs for the related model
    link_df = link_df.join(
        related_model_df.rename({"external_id": actual_field_name}),
        on=actual_field_name,
        how="inner",
    ).rename({"db_id": f"{related_model_fk}/id"})

    # 4. Write to a temporary file and return import details
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".csv", newline=""
    ) as tmp:
        link_df.select([owning_model_fk, f"{related_model_fk}/id"]).write_csv(tmp.name)
        tmp_path = tmp.name

    return {
        "file_csv": tmp_path,
        "model": relational_table,
        "unique_id_field": owning_model_fk,
    }


def _prepare_link_dataframe(
    source_df: pl.DataFrame,
    actual_field_name: str,
    owning_df: pl.DataFrame,
    related_model_df: pl.DataFrame,
    owning_model_fk: str,
    related_model_fk: str,
) -> pl.DataFrame:
    """Prepare the link table DataFrame for relational imports.

    Args:
        source_df: The source DataFrame
        actual_field_name: The actual field name in the DataFrame
            (may include /id suffix)
        owning_df: DataFrame with owning model IDs
        related_model_df: DataFrame with related model IDs
        owning_model_fk: The owning model foreign key field name
        related_model_fk: The related model name

    Returns:
        The prepared link DataFrame
    """
    # Debug: Print available columns and the field we're looking for
    log.debug(f"Available columns in source_df: {source_df.columns}")
    log.debug(f"Looking for field: {actual_field_name}")

    # Check if the field exists in the DataFrame
    if actual_field_name not in source_df.columns:
        log.error(
            f"Field '{actual_field_name}' not found in source DataFrame. "
            f"Available columns: {source_df.columns}"
        )
        # Return an empty DataFrame with the expected schema
        return pl.DataFrame(
            schema={
                "external_id": pl.Utf8,
                actual_field_name: pl.Utf8,
                owning_model_fk: pl.Int64,
                f"{related_model_fk}/id": pl.Int64,
            }
        )

    # Create the link table DataFrame
    link_df = source_df.select(["id", actual_field_name]).rename({"id": "external_id"})
    link_df = link_df.with_columns(pl.col(actual_field_name).str.split(",")).explode(
        actual_field_name
    )

    # Join to get DB IDs for the owning model
    link_df = link_df.join(owning_df, on="external_id", how="inner").rename(
        {"db_id": owning_model_fk}
    )

    # Join to get DB IDs for the related model
    link_df = link_df.join(
        related_model_df.rename({"external_id": actual_field_name}),
        on=actual_field_name,
        how="inner",
    ).rename({"db_id": f"{related_model_fk}/id"})

    return link_df


def run_write_tuple_import(
    config: Union[str, dict[str, Any]],
    model: str,
    field: str,
    strategy_details: dict[str, Any],
    source_df: pl.DataFrame,
    id_map: dict[str, int],
    worker: int,
    batch_size: int,
    progress: Progress,
    task_id: TaskID,
    original_filename: str,
) -> bool:
    """Orchestrates the 'write_tuple' import for relational fields."""
    progress.update(
        task_id,
        description=f"Pass 2/2: Updating relations for [bold]{field}[/bold]",
    )
    log.info(f"*** RUNNING WRITE TUPLE IMPORT FOR FIELD '{field}' ***")
    log.info(f"*** STRATEGY DETAILS: {strategy_details} ***")

    # Add a small delay to reduce server load and prevent connection pool exhaustion
    import time

    time.sleep(0.1)

    # Check if required keys exist
    relational_table = strategy_details.get("relation_table")
    owning_model_fk = strategy_details.get("relation_field")
    related_model_fk = strategy_details.get("relation")

    log.info(f"*** RELATIONAL TABLE: {relational_table} ***")
    log.info(f"*** OWNING MODEL FK: {owning_model_fk} ***")
    log.info(f"*** RELATED MODEL FK: {related_model_fk} ***")

    # Try to derive missing information if possible
    relational_table, owning_model_fk = _derive_missing_relation_info(
        config, model, field, relational_table, owning_model_fk, related_model_fk
    )

    # If we still don't have the required information, we can't proceed
    # with this strategy
    if not relational_table or not owning_model_fk:
        log.error(
            f"Cannot run write tuple import for field '{field}': "
            f"Missing relation_table or relation_field in strategy details."
        )
        return False

    # 1. Prepare the owning model's IDs
    # Normalize external IDs to ensure consistency between pass 1 and pass 2
    # This addresses the join issue where external IDs didn't match exactly
    normalized_id_map = {
        str(external_id).strip(): db_id for external_id, db_id in id_map.items()
    }
    owning_df = pl.DataFrame(
        {
            "external_id": list(normalized_id_map.keys()),
            "db_id": list(normalized_id_map.values()),
        },
        schema={"external_id": pl.Utf8, "db_id": pl.Int64},
    )

    # Debug: Print available columns and the field we're looking for
    log.debug(f"Available columns in source_df: {source_df.columns}")
    log.debug(f"Looking for field: {field}")
    log.debug(f"Field '{field}' in source_df.columns: {field in source_df.columns}")

    # Check if the field exists in the DataFrame (redundant check for debugging)
    if field not in source_df.columns:
        # Check if the field with /id suffix exists (common for relation fields)
        field_with_id = f"{field}/id"
        if field_with_id in source_df.columns:
            log.debug(f"Using field '{field_with_id}' instead of '{field}'")
            field = field_with_id
        else:
            log.error(
                f"Field '{field}' not found in source DataFrame. "
                f"Available columns: {source_df.columns}"
            )
            return False

    # 2. Prepare the related model's IDs using the resolver
    all_related_ext_ids = source_df.get_column(field).str.split(",").explode()
    log.info(
        "*** TOTAL RELATED EXTERNAL IDS BEFORE FILTERING: "
        f"{len(all_related_ext_ids)} ***"
    )
    log.info(
        f"*** SAMPLE RELATED EXTERNAL IDS: {all_related_ext_ids.head(5).to_list()} ***"
    )
    if related_model_fk is None:
        log.error(
            f"Cannot resolve related IDs: Missing relation in strategy details "
            f"for field '{field}'."
        )
        return False
    related_model_df = _resolve_related_ids(
        config, related_model_fk, all_related_ext_ids
    )
    if related_model_df is None:
        log.error(f"Could not resolve IDs for related model '{related_model_fk}'.")
        return False
    log.info(f"*** RELATED MODEL DF SHAPE: {related_model_df.shape} ***")
    log.info(f"*** RELATED MODEL DF SAMPLE: {related_model_df.head(3)} ***")

    # 3. Create the link table DataFrame
    # Ensure external_id column is properly typed and normalized
    link_df = source_df.select(["id", field]).rename({"id": "external_id"})
    # Normalize external IDs to match the format used in id_map
    link_df = link_df.with_columns(
        [pl.col("external_id").cast(pl.Utf8).str.strip_chars()]
    )
    link_df = link_df.with_columns(pl.col(field).str.split(",")).explode(field)

    log.info(f"*** LINK DF SHAPE BEFORE OWNING JOIN: {link_df.shape} ***")
    log.info(f"*** LINK DF SAMPLE BEFORE OWNING JOIN: {link_df.head(3)} ***")

    link_df = link_df.join(owning_df, on="external_id", how="inner").rename(
        {"db_id": owning_model_fk}
    )
    log.info(f"*** LINK DF SHAPE AFTER OWNING JOIN: {link_df.shape} ***")
    log.info(f"*** LINK DF SAMPLE AFTER OWNING JOIN: {link_df.head(3)} ***")

    # Join to get DB IDs for the related model
    link_df = link_df.join(
        related_model_df.rename({"external_id": field}), on=field, how="inner"
    ).rename({"db_id": f"{related_model_fk}/id"})
    log.info(f"*** LINK DF SHAPE AFTER RELATED JOIN: {link_df.shape} ***")
    log.info(f"*** LINK DF SAMPLE AFTER RELATED JOIN: {link_df.head(3)} ***")

    # 4. Execute the updates
    success = _execute_write_tuple_updates(
        config, model, field, link_df, id_map, related_model_fk, original_filename
    )

    # Count successful updates - get from link_df
    if link_df.height > 0:
        successful_count = len(
            [
                row["external_id"]
                for row in link_df.iter_rows(named=True)
                if id_map.get(row["external_id"])
            ]
        )
    else:
        successful_count = 0
    failed_count = 0 if success else "unknown"

    log.info(
        f"Finished 'Write Tuple' for '{field}': "
        f"{successful_count} successful, {failed_count} failed."
    )

    return success


def _create_relational_records(
    config: Union[str, dict[str, Any]],
    model: str,
    field: str,
    actual_field_name: str,
    relational_table: str,
    owning_model_fk: str,
    related_model_fk: str,
    link_df: pl.DataFrame,
    owning_df: pl.DataFrame,
    related_model_df: pl.DataFrame,
    original_filename: str,
    batch_size: int,
) -> bool:
    """Create records in the relational table.

    For many2many relationships in Odoo, we need to update the owning model's
    field with special commands, rather than trying to access the relationship
    table directly as a model.

    Args:
        config: Configuration for the connection
        model: The model name (owning model)
        field: The field name (many2many field)
        actual_field_name: The actual field name in the DataFrame
            (may include /id suffix)
        relational_table: The relational table name
        owning_model_fk: The owning model foreign key field name
        related_model_fk: The related model name
        link_df: The link DataFrame
        owning_df: DataFrame with owning model IDs
        related_model_df: DataFrame with related model IDs
        original_filename: The original filename
        batch_size: The batch size for processing

    Returns:
        True if successful, False otherwise
    """
    if isinstance(config, dict):
        connection = conf_lib.get_connection_from_dict(config)
    else:
        connection = conf_lib.get_connection_from_config(config_file=config)

    # For many2many relationships, we need to use the owning model to set the field
    # rather than trying to access the relationship table directly as a model
    try:
        owning_model = connection.get_model(model)
    except Exception as e:
        log.error(f"Failed to access owning model '{model}' in Odoo. Error: {e}")
        return False

    # We need to map back to the original external IDs for failure reporting
    # This is a bit heavy, but necessary for accurate error logs.
    # The link_df contains the external_id column and the actual field column
    # These columns already contain individual IDs (not comma-separated) because
    # they have been processed by _prepare_link_dataframe
    original_links_df = link_df.select(["external_id", actual_field_name]).rename(
        {"external_id": "parent_external_id", actual_field_name: "related_external_id"}
    )

    # Join with resolved IDs to get the data for updating records
    update_df = original_links_df.join(
        owning_df.rename({"external_id": "parent_external_id"}),
        on="parent_external_id",
        how="inner",
    ).rename({"db_id": owning_model_fk})
    update_df = update_df.join(
        related_model_df.rename({"external_id": "related_external_id"}),
        on="related_external_id",
        how="inner",
    ).rename({"db_id": f"{related_model_fk}/id"})

    # Group by owning model ID and collect all related IDs for each owner
    # This is needed because we update each owning record once with all
    # its related records
    # Use Polars group_by and agg for better performance than row iteration
    grouped_df = update_df.group_by(owning_model_fk).agg(
        pl.col(f"{related_model_fk}/id")
    )
    # Convert Polars Series to Python lists for type safety
    grouped_data: dict[int, list[int]] = {}
    for i in range(len(grouped_df)):
        owning_id = grouped_df[owning_model_fk][i]
        related_ids_series = grouped_df[f"{related_model_fk}/id"][i]
        grouped_data[owning_id] = related_ids_series.to_list()

    successful_updates = 0
    failed_records_to_report = []

    # Update each owning record with its many2many field values
    for owning_id, related_ids in grouped_data.items():
        try:
            # For many2many fields, we use the (6, 0, [IDs]) command to replace
            # the entire set of related records for this owner
            # This replaces any existing relationships with the new set
            m2m_command = [(6, 0, related_ids)]

            # Update the owning record with the many2many field
            owning_model.write([owning_id], {field: m2m_command})
            successful_updates += 1

        except Exception as e:
            log.error(
                f"Failed to update record {owning_id} with many2many field '{field}'. "
                f"Reason: {e}"
            )
            # Find the corresponding report items and add them to failed records
            failed_items = [
                {
                    "model": model,
                    "field": field,
                    "parent_external_id": row["parent_external_id"],
                    "related_external_id": row["related_external_id"],
                    "error_reason": str(e),
                }
                for row in update_df.filter(
                    pl.col(owning_model_fk) == owning_id
                ).iter_rows(named=True)
            ]
            failed_records_to_report.extend(failed_items)

    if failed_records_to_report:
        writer.write_relational_failures_to_csv(
            model, field, original_filename, failed_records_to_report
        )

    failed_updates = len(failed_records_to_report)
    log.info(
        f"Finished 'Write Tuple' for '{field}': "
        f"{successful_updates} successful, {failed_updates} failed."
    )

    return successful_updates > 0


def run_write_o2m_tuple_import(
    config: Union[str, dict[str, Any]],
    model: str,
    field: str,
    strategy_details: dict[str, Any],
    source_df: pl.DataFrame,
    id_map: dict[str, int],
    worker: int,
    batch_size: int,
    progress: Progress,
    task_id: TaskID,
    original_filename: str,
) -> bool:
    """Orchestrates the 'write_o2m_tuple' import for one2many fields."""
    progress.update(
        task_id,
        description=f"Pass 2/2: Updating relations for [bold]{field}[/bold]",
    )
    log.info(f"Running 'Write O2M Tuple' for field '{field}'...")

    if isinstance(config, dict):
        connection = conf_lib.get_connection_from_dict(config)
    else:
        connection = conf_lib.get_connection_from_config(config_file=config)
    parent_model = connection.get_model(model)
    successful_updates = 0
    failed_records_to_report = []

    # Filter for rows that actually have data in the o2m field
    # Check if the field with /id suffix exists (common for relation fields)
    actual_field_name = field
    if f"{field}/id" in source_df.columns:
        actual_field_name = f"{field}/id"

    o2m_df = source_df.filter(pl.col(actual_field_name).is_not_null())

    for record in o2m_df.iter_rows(named=True):
        parent_external_id = record["id"]
        parent_db_id = id_map.get(parent_external_id)
        if not parent_db_id:
            continue

        o2m_json_data = record[actual_field_name]
        try:
            child_records = json.loads(o2m_json_data)
            if not isinstance(child_records, list):
                raise ValueError("JSON data is not a list")

            # Odoo command: (0, 0, {values}) for creating new records
            o2m_commands = [(0, 0, vals) for vals in child_records]
            parent_model.write([parent_db_id], {field: o2m_commands})
            successful_updates += 1

        except json.JSONDecodeError:
            log.error(
                f"Failed to decode JSON for parent '{parent_external_id}' "
                f"in field '{field}'. Value: {o2m_json_data}"
            )
            failed_records_to_report.append(
                {
                    "model": model,
                    "field": field,
                    "parent_external_id": parent_external_id,
                    "related_external_id": "N/A (JSON Data)",
                    "error_reason": "Invalid JSON format",
                }
            )
        except Exception as e:
            log.error(
                f"Failed to write o2m commands for parent '{parent_external_id}': {e}"
            )
            failed_records_to_report.append(
                {
                    "model": model,
                    "field": field,
                    "parent_external_id": parent_external_id,
                    "related_external_id": "N/A (JSON Data)",
                    "error_reason": str(e),
                }
            )

    if failed_records_to_report:
        writer.write_relational_failures_to_csv(
            model, field, original_filename, failed_records_to_report
        )

    log.info(
        f"Finished 'Write O2M Tuple' for '{field}': "
        f"{successful_updates} successful, {len(failed_records_to_report)} failed."
    )
    return successful_updates > 0
