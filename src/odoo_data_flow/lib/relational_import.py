"""Handles relational import strategies like m2m and o2m."""

import json
import tempfile
from typing import Any, Optional

import polars as pl
from rich.progress import Progress, TaskID

from ..logging_config import log
from . import cache, conf_lib, writer


def _resolve_related_ids(
    config: str, related_model: str, external_ids: pl.Series
) -> Optional[pl.DataFrame]:
    """Resolve related ids.

    Resolves external IDs for a related model, trying cache first,
    then falling back to a bulk XML-ID resolution.
    """
    # 1. Try to load from cache
    related_model_cache = cache.load_id_map(config, related_model)
    if related_model_cache is not None:
        log.info(f"Cache hit for related model '{related_model}'.")
        return related_model_cache

    # 2. Fallback to bulk XML-ID resolution
    log.warning(
        f"Cache miss for related model '{related_model}'. "
        f"Falling back to slow XML-ID resolution."
    )
    connection = conf_lib.get_connection_from_config(config_file=config)
    if not connection.is_connected():
        log.error("Cannot perform XML-ID lookup: Odoo connection failed.")
        return None

    id_list = external_ids.drop_nulls().unique().to_list()
    log.info(f"Resolving {len(id_list)} unique external IDs for '{related_model}'...")

    # Split full XML-ID 'module.identifier' into components
    split_ids = [(i.split(".", 1)[0], i.split(".", 1)[1]) for i in id_list if "." in i]
    invalid_ids = [i for i in id_list if "." not in i]
    if invalid_ids:
        log.warning(
            f"Skipping {len(invalid_ids)} invalid external_ids for model "
            f"'{related_model}' (must be in 'module.identifier' format)."
        )

    domain = [
        "&",
        ("module", "=", split_ids[0][0]),
        ("name", "=", split_ids[0][1]),
    ]
    for module, name in split_ids[1:]:
        domain.insert(0, "|")
        domain.append("&")
        domain.append(("module", "=", module))
        domain.append(("name", "=", name))

    try:
        data_model = connection.get_model("ir.model.data")
        resolved_data = data_model.search_read(domain, ["module", "name", "res_id"])
        if not resolved_data:
            log.error(
                f"XML-ID resolution failed for all IDs in model '{related_model}'."
            )
            return None

        resolved_map = {
            f"{rec['module']}.{rec['name']}": rec["res_id"] for rec in resolved_data
        }

        log.info(
            f"Successfully resolved {len(resolved_map)} out of {len(id_list)} "
            f"external IDs for model '{related_model}'."
        )
        return pl.DataFrame(
            {"external_id": resolved_map.keys(), "db_id": resolved_map.values()}
        )
    except Exception as e:
        log.error(f"An error occurred during bulk XML-ID resolution: {e}")
        return None


def run_direct_relational_import(
    config: str,
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
    relational_table = strategy_details["relation_table"]
    owning_model_fk = strategy_details["relation_field"]
    related_model_fk = strategy_details["relation"]

    # 1. Prepare the owning model's IDs
    owning_df = pl.DataFrame({"external_id": id_map.keys(), "db_id": id_map.values()})

    # 2. Prepare the related model's IDs using the resolver
    all_related_ext_ids = source_df.get_column(field).str.split(",").explode()
    related_model_df = _resolve_related_ids(
        config, related_model_fk, all_related_ext_ids
    )
    if related_model_df is None:
        log.error(f"Could not resolve IDs for related model '{related_model_fk}'.")
        return None

    # 3. Create the link table DataFrame
    link_df = source_df.select(["id", field]).rename({"id": "external_id"})
    link_df = link_df.with_columns(pl.col(field).str.split(",")).explode(field)

    # Join to get DB IDs for the owning model
    link_df = link_df.join(owning_df, on="external_id", how="inner").rename(
        {"db_id": owning_model_fk}
    )

    # Join to get DB IDs for the related model
    link_df = link_df.join(
        related_model_df.rename({"external_id": field}), on=field, how="inner"
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


def run_write_tuple_import(
    config: str,
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
    log.info(f"Running 'Write Tuple' for field '{field}'...")
    relational_table = strategy_details["relation_table"]
    owning_model_fk = strategy_details["relation_field"]
    related_model_fk = strategy_details["relation"]

    # 1. Prepare the owning model's IDs
    owning_df = pl.DataFrame({"external_id": id_map.keys(), "db_id": id_map.values()})

    # 2. Prepare the related model's IDs using the resolver
    all_related_ext_ids = source_df.get_column(field).str.split(",").explode()
    related_model_df = _resolve_related_ids(
        config, related_model_fk, all_related_ext_ids
    )
    if related_model_df is None:
        log.error(f"Could not resolve IDs for related model '{related_model_fk}'.")
        return False

    # 3. Create the link table DataFrame
    link_df = source_df.select(["id", field]).rename({"id": "external_id"})
    link_df = link_df.with_columns(pl.col(field).str.split(",")).explode(field)

    # Join to get DB IDs for the owning model
    link_df = link_df.join(owning_df, on="external_id", how="inner").rename(
        {"db_id": owning_model_fk}
    )

    # Join to get DB IDs for the related model
    link_df = link_df.join(
        related_model_df.rename({"external_id": field}), on=field, how="inner"
    ).rename({"db_id": f"{related_model_fk}/id"})

    # 4. Create records in the relational table
    connection = conf_lib.get_connection_from_config(config_file=config)
    rel_model = connection.get_model(relational_table)

    # We need to map back to the original external IDs for failure reporting
    # This is a bit heavy, but necessary for accurate error logs.
    original_links_df = source_df.select(["id", field]).rename(
        {"id": "parent_external_id"}
    )
    original_links_df = original_links_df.with_columns(
        pl.col(field).str.split(",")
    ).explode(field)
    original_links_df = original_links_df.rename({field: "related_external_id"})

    # Join with resolved IDs to get the data for `create`
    create_df = original_links_df.join(
        owning_df.rename({"external_id": "parent_external_id"}),
        on="parent_external_id",
        how="inner",
    ).rename({"db_id": owning_model_fk})
    create_df = create_df.join(
        related_model_df.rename({"external_id": "related_external_id"}),
        on="related_external_id",
        how="inner",
    ).rename({"db_id": f"{related_model_fk}/id"})

    vals_list = create_df.select([owning_model_fk, f"{related_model_fk}/id"]).to_dicts()
    # Keep original IDs for error reporting
    report_list = create_df.select(
        ["parent_external_id", "related_external_id"]
    ).to_dicts()

    successful_creates = 0
    failed_records_to_report = []
    batch_size = 50

    for i in range(0, len(vals_list), batch_size):
        vals_batch = vals_list[i : i + batch_size]
        report_batch = report_list[i : i + batch_size]
        try:
            rel_model.create(vals_batch)
            successful_creates += len(vals_batch)
        except Exception as e:
            log.error(
                f"Failed to create a batch of {len(vals_batch)} records for "
                f"'{relational_table}'. Reason: {e}"
            )
            # Fallback to one-by-one to salvage what we can and log failures
            for j, vals in enumerate(vals_batch):
                try:
                    rel_model.create(vals)
                    successful_creates += 1
                except Exception as inner_e:
                    report_item = report_batch[j]
                    report_item["model"] = model
                    report_item["field"] = field
                    report_item["error_reason"] = str(inner_e)
                    failed_records_to_report.append(report_item)

    if failed_records_to_report:
        writer.write_relational_failures_to_csv(
            model, field, original_filename, failed_records_to_report
        )

    failed_creates = len(failed_records_to_report)
    log.info(
        f"Finished 'Write Tuple' for '{field}': "
        f"{successful_creates} successful, {failed_creates} failed."
    )

    return successful_creates > 0


def run_write_o2m_tuple_import(
    config: str,
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

    connection = conf_lib.get_connection_from_config(config_file=config)
    parent_model = connection.get_model(model)
    successful_updates = 0
    failed_records_to_report = []

    # Filter for rows that actually have data in the o2m field
    o2m_df = source_df.filter(pl.col(field).is_not_null())

    for record in o2m_df.iter_rows(named=True):
        parent_external_id = record["id"]
        parent_db_id = id_map.get(parent_external_id)
        if not parent_db_id:
            continue

        o2m_json_data = record[field]
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
