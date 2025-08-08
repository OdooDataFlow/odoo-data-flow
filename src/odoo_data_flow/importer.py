"""Main importer module.

This module contains the high-level logic for orchestrating the import process.
It handles file I/O, pre-flight checks, and the delegation of the core
import tasks to the multi-threaded `import_threaded` module.
"""

import ast
import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, cast

from rich.console import Console
from rich.panel import Panel

from . import import_threaded
from .enums import PreflightMode
from .lib import preflight
from .lib.internal.ui import _show_error_panel
from .logging_config import log


def _get_fail_filename(model: str, is_fail_run: bool) -> str:
    """Generates a standardized filename for failed records."""
    model_filename = model.replace(".", "_")
    if is_fail_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_filename}_{timestamp}_failed.csv"
    return f"{model_filename}_fail.csv"


# --- MODIFIED: Accept and pass the import_plan dictionary ---
def _run_preflight_checks(
    preflight_mode: PreflightMode, import_plan: dict[str, Any], **kwargs: Any
) -> bool:
    """Iterates through and runs all registered pre-flight checks."""
    for check_func in preflight.PREFLIGHT_CHECKS:
        if not check_func(
            preflight_mode=preflight_mode, import_plan=import_plan, **kwargs
        ):
            return False
    return True


def run_import(  # noqa: C901
    config: str,
    filename: str,
    model: Optional[str] = None,
    deferred_fields: Optional[str] = None,
    unique_id_field: Optional[str] = None,
    no_preflight_checks: bool = False,
    headless: bool = False,
    worker: int = 1,
    batch_size: int = 10,
    skip: int = 0,
    fail: bool = False,
    separator: str = ";",
    split: Optional[Union[str, tuple[str, ...], list[str]]] = None,
    split_by_cols: Optional[Union[str, tuple[str, ...], list[str]]] = None,
    ignore: Optional[str] = None,
    check: bool = False,
    context: str = "{'tracking_disable' : True}",
    o2m: bool = False,
    encoding: str = "utf-8",
    field_module_map_file: Optional[str] = None,
) -> None:
    """Orchestrates the data import process from a CSV file.

    Args:
        config: Path to the connection configuration file.
        filename: Path to the source CSV file to import.
        model: The Odoo model to import data into. If not provided, it's inferred
               from the filename.
        deferred_fields: A comma-separated string of field names to defer to a
                        second pass. Triggers the two-pass import strategy.
        unique_id_field: The name of the column in the CSV that uniquely
                        identifies each record. Required for deferred imports.
        no_preflight_checks: If True, skips all pre-flight validation checks.
        headless: If True, runs in non-interactive mode, auto-confirming any
                  prompts (e.g., installing languages).
        worker: The number of simultaneous connections to use.
        batch_size: The number of records to process in each batch.
        skip: The number of initial lines to skip in the source file.
        fail: If True, runs in fail mode, retrying records from the _fail.csv file.
        separator: The delimiter used in the CSV file.
        split: [DEPRECATED] Use `split_by_cols` instead. Kept for backward
               compatibility with internal tests.
        split_by_cols: A column name (string), or a list/tuple of column
                       names to group records by to avoid concurrent updates on
                       the same parent record.
        ignore: A comma-separated string of column names to ignore.
        check: If True, checks if records were successfully imported.
        context: A string representation of the Odoo context dictionary.
        o2m: If True, enables special handling for one-to-many imports.
        encoding: The file encoding of the source file.
        field_module_map_file: Path to the CSV linking fields to module to
            suggest for installation.
    """
    log.info("Starting data import process from file...")

    # --- 1. Initial Setup and Model Inference ---
    final_model = model
    if not final_model:
        base_name = os.path.basename(filename)
        inferred_model = os.path.splitext(base_name)[0].replace("_", ".")
        if not inferred_model or inferred_model.startswith("."):
            _show_error_panel(
                "Model Not Found",
                f"Could not infer model from filename '{base_name}'. "
                f"Please use the --model option.",
            )
            return
        final_model = inferred_model
        log.info(f"No model provided. Inferred model '{final_model}' from filename.")

    if fail and deferred_fields:
        _show_error_panel(
            "Invalid Arguments",
            "The --fail flag cannot be used with --deferred-fields at the same time.",
        )
        return

    # --- 2. Handle Fail Mode and Determine File to Process ---
    current_preflight_mode = PreflightMode.NORMAL
    file_to_process = filename
    if fail:
        current_preflight_mode = PreflightMode.FAIL_MODE
        fail_filename = _get_fail_filename(final_model, is_fail_run=False)
        fail_file_path = Path(filename).parent / fail_filename

        file_has_records_to_retry = False
        if fail_file_path.exists():
            with open(fail_file_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                try:
                    next(reader)  # Header
                    next(reader)  # First data row
                    file_has_records_to_retry = True
                except StopIteration:
                    file_has_records_to_retry = False

        if not file_has_records_to_retry:
            Console().print(
                Panel(
                    f"No records found in '{fail_file_path}'. Nothing to retry.",
                    title="[bold green]No Recovery Needed[/bold green]",
                    border_style="green",
                )
            )
            return  # EXIT EARLY before pre-flight checks

        log.info(f"Running in --fail mode. Retrying records from: {fail_file_path}")
        file_to_process = str(fail_file_path)

    # --- 3. Pre-flight Checks ---
    import_plan: dict[str, Any] = {}
    if not no_preflight_checks:
        if not _run_preflight_checks(
            preflight_mode=current_preflight_mode,
            import_plan=import_plan,
            model=final_model,
            filename=file_to_process,  # Use the correctly determined file
            config=config,
            headless=headless,
            separator=separator,
            unique_id_field=unique_id_field,
            field_module_map_file=field_module_map_file,  # <-- NEW ARG
        ):
            return

    # --- 4. Select Import Strategy (Routing) ---
    if not fail:
        auto_detected_deferred_fields = import_plan.get("deferred_fields", [])
        if deferred_fields and unique_id_field:
            log.info(
                "User has specified deferred fields. Using two-pass import strategy."
            )
            deferred_list = [field.strip() for field in deferred_fields.split(",")]

        # Strategy 1: User explicitly requests deferred import
        if deferred_fields and unique_id_field:
            log.info(
                "User has specified deferred fields. Using two-pass import strategy."
            )
            deferred_list = [field.strip() for field in deferred_fields.split(",")]
            run_import_deferred(
                config=config,
                filename=filename,
                model_name=final_model,
                unique_id_field=unique_id_field,
                deferred_fields=deferred_list,
                encoding=encoding,
                separator=separator,
            )
            return

        # Strategy 2: Automatic detection triggers deferred import
        final_unique_id_field = unique_id_field or import_plan.get("unique_id_field")
        if auto_detected_deferred_fields and final_unique_id_field:
            log.info(
                f"Auto-detected deferrable fields: {auto_detected_deferred_fields}. "
                "Switching to two-pass import strategy."
            )
            run_import_deferred(
                config=config,
                filename=filename,
                model_name=final_model,
                unique_id_field=final_unique_id_field,
                deferred_fields=auto_detected_deferred_fields,
                encoding=encoding,
                separator=separator,
            )
            return

    # --- 5. Execute Standard Single-Pass Import (Fallback) ---
    log.info("No deferrable fields specified or detected. Using standard import.")
    try:
        parsed_context = ast.literal_eval(context)
        if not isinstance(parsed_context, dict):
            raise TypeError("Context must be a dictionary.")
    except Exception as e:
        _show_error_panel("Invalid Context", f"Invalid --context dictionary: {e}")
        return

    ignore_list = ignore.split(",") if ignore else []
    file_dir = os.path.dirname(filename)
    model_filename_part = final_model.replace(".", "_")

    if fail:
        force_create_run = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fail_output_file = os.path.join(
            file_dir, f"{model_filename_part}_{timestamp}_failed.csv"
        )
        batch_size_run, max_connection_run = 1, 1
    else:
        force_create_run = False
        fail_output_file = os.path.join(file_dir, f"{model_filename_part}_fail.csv")
        batch_size_run, max_connection_run = int(batch_size), int(worker)

    log.info(f"Importing file: {file_to_process}")
    log.info(f"Target model: {final_model}")
    log.info(f"Workers: {max_connection_run}, Batch Size: {batch_size_run}")

    success = import_threaded.import_data(
        config_file=config,
        model=final_model,
        unique_id_field=unique_id_field or "id",  # Default to 'id' for standard runs
        file_csv=file_to_process,
        context=parsed_context,
        fail_file=fail_output_file,
        encoding=encoding,
        separator=separator,
        ignore=ignore_list,
        max_connection=max_connection_run,
        batch_size=batch_size_run,
        skip=int(skip),
        force_create=force_create_run,
    )

    console = Console()
    if success:
        console.print(
            Panel(
                f"Import for [cyan]{final_model}[/cyan] finished successfully.",
                title="[bold green]Import Complete[/bold green]",
                border_style="green",
            )
        )
    else:
        _show_error_panel(
            "Import Aborted", "The import was aborted due to a critical error."
        )


def run_import_for_migration(
    config: str,
    model: str,
    header: list[str],
    data: list[list[Any]],
    worker: int = 1,
    batch_size: int = 10,
) -> None:
    """Orchestrates the data import process from in-memory data.

    Args:
        config: Path to the connection configuration file.
        model: The Odoo model to import data into.
        header: A list of strings representing the column headers.
        data: A list of lists representing the data rows.
        worker: The number of simultaneous connections to use.
        batch_size: The number of records to process in each batch.
    """
    log.info("Starting data import from in-memory data...")

    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".csv", newline=""
    ) as tmp:
        writer = csv.writer(tmp)
        writer.writerow(header)
        writer.writerows(data)
        tmp_path = tmp.name

    log.info(f"In-memory data written to temporary file: {tmp_path}")

    try:
        import_threaded.import_data(
            config_file=config,
            model=model,
            unique_id_field="id",  # Migration import assumes 'id'
            file_csv=tmp_path,
            context={"tracking_disable": True},
            max_connection=int(worker),
            batch_size=int(batch_size),
        )
    finally:
        os.remove(tmp_path)  # Clean up the temporary file

    log.info("In-memory import process finished.")


def _execute_pass_1(
    model: Any,
    records: list[dict[str, Any]],
    deferred_fields: list[str],
    unique_id_field: str,
) -> dict[str, int]:
    """Prepares and executes the first pass (creation) of a deferred import.

    This function takes the full list of records, removes the deferred fields,
    and calls the batch_create method to create the base records in Odoo.

    Args:
        model: The Odoo model object with a `batch_create` method.
        records: A list of dictionaries representing the source data.
        deferred_fields: A list of field names to exclude from this pass.
        unique_id_field: The key in the record dictionaries used for mapping.

    Returns:
        A dictionary mapping the unique source ID from `unique_id_field` to
        the newly created Odoo database ID.
    """
    log.info("Executing Pass 1: Creating base records...")
    pass_1_data = [
        {k: v for k, v in record.items() if k not in deferred_fields}
        for record in records
    ]

    id_map_any = model.batch_create(pass_1_data, unique_id_field)
    id_map = cast(dict[str, int], id_map_any)
    log.info(f"Pass 1 complete. Created {len(id_map)} records.")
    return id_map


def _execute_pass_2(
    model: Any,
    records: list[dict[str, Any]],
    deferred_fields: list[str],
    unique_id_field: str,
    id_map: dict[str, int],
) -> int:
    """Prepares and executes the second pass (update) of a deferred import.

    This function iterates through the original records and uses the `id_map`
    from Pass 1 to build and execute write payloads for the deferred fields.

    Args:
        model: The Odoo model object with a `batch_write` method.
        records: The original list of dictionaries representing the source data.
        deferred_fields: The list of fields to update in this pass.
        unique_id_field: The key in the record dictionaries for mapping.
        id_map: The map of source IDs to database IDs returned by Pass 1.

    Returns:
        An integer count of the records that were successfully updated.
    """
    log.info("Executing Pass 2: Updating records with deferred relations...")
    pass_2_data = []
    for record in records:
        unique_id = record.get(unique_id_field)
        if not unique_id:
            log.warning("Skipping update for a record with no unique ID.")
            continue

        record_db_id = id_map.get(unique_id)
        if not record_db_id:
            log.warning("Skipping update for '%s'; not created in Pass 1.", unique_id)
            continue

        update_vals = {}
        for field in deferred_fields:
            source_relation_id = record.get(field)
            if source_relation_id and id_map.get(source_relation_id):
                update_vals[field] = id_map[source_relation_id]
            elif source_relation_id:
                log.warning(
                    "Could not resolve relation for ID '%s' in field '%s'.",
                    source_relation_id,
                    field,
                )

        if update_vals:
            pass_2_data.append((record_db_id, update_vals))

    if not pass_2_data:
        log.info("No deferred relations found to update; skipping Pass 2.")
        return 0

    write_summary_any = model.batch_write(pass_2_data)
    write_summary = cast(dict[str, int], write_summary_any)
    updates_made = write_summary.get("success", 0)
    log.info(f"Pass 2 complete. Updated {updates_made} records.")
    return updates_made


def run_import_deferred(
    config: str,
    filename: str,
    model_name: str,
    unique_id_field: str,
    deferred_fields: list[str],
    encoding: str = "utf-8",
    separator: str = ";",
) -> bool:
    """Performs a two-pass import from a CSV file to handle deferred relations.

    Args:
        config: Path to the connection configuration file.
        filename: Path to the source CSV file.
        model_name: The technical name of the Odoo model (e.g., 'res.partner').
        unique_id_field: The column in the CSV that uniquely identifies each row.
        deferred_fields: A list of column names to exclude from the first pass.
        encoding: The file encoding of the source file.
        separator: The delimiter used in the CSV file.

    Returns:
        True if the import process completes successfully, False otherwise.
    """
    log.info(f"Starting two-pass deferred import for model '{model_name}'...")

    # This function now acts as a simple wrapper that calls the powerful
    # low-level import engine with the correct parameters.
    success = import_threaded.import_data(
        config_file=config,
        model=model_name,
        unique_id_field=unique_id_field,
        file_csv=filename,
        deferred_fields=deferred_fields,
        encoding=encoding,
        separator=separator,
        # Sensible defaults for a deferred run
        max_connection=4,  # Allow multi-threading
        batch_size=200,
    )

    console = Console()
    if success:
        console.print(
            Panel(
                f"Two-pass import for [cyan]{model_name}[/cyan] finished.",
                title="[bold green]Import Complete[/bold green]",
                expand=False,
            )
        )
    else:
        _show_error_panel(
            "Import Failed",
            "The deferred import process failed. Check logs for details.",
        )

    return success
