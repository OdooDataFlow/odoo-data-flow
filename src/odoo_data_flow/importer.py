"""Main importer module.

This module contains the high-level logic for orchestrating the import process.
It handles file I/O, pre-flight checks, and the delegation of the core
import tasks to the multi-threaded `import_threaded` module.
"""

import csv
import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, cast

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

from . import import_threaded
from .enums import PreflightMode
from .lib import cache, preflight, relational_import, sort
from .lib.internal.ui import _show_error_panel
from .logging_config import log


def _map_encoding_to_polars(encoding: str) -> str:
    """Map common encoding names to polars-supported encoding values.

    Polars supports: 'utf8', 'utf8-lossy', 'windows-1252', 'windows-1252-lossy'
    This function maps common encoding names to these supported values.

    Args:
        encoding: The encoding name to map

    Returns:
        A polars-supported encoding name
    """
    # Normalize encoding names to lowercase
    encoding = encoding.lower().strip()

    # Mapping for common encoding names to polars-supported values
    encoding_map = {
        # UTF variants
        "utf-8": "utf8",
        "utf8": "utf8",
        "utf-8-sig": "utf8",  # UTF-8 with BOM
        # Latin variants commonly used in Western Europe
        "latin-1": "windows-1252",
        "iso-8859-1": "windows-1252",
        "cp1252": "windows-1252",
        "windows-1252": "windows-1252",
        # Lossy variants - when we want to preserve as much data as possible
        "utf-8-lossy": "utf8-lossy",
        "latin-1-lossy": "windows-1252-lossy",
        "iso-8859-1-lossy": "windows-1252-lossy",
        "cp1252-lossy": "windows-1252-lossy",
        "windows-1252-lossy": "windows-1252-lossy",
    }

    # Return mapped encoding if available, otherwise return the original
    # (will be validated by polars)
    return encoding_map.get(encoding, encoding)


def _count_lines(filepath: str) -> int:
    """Counts the number of lines in a file, returning 0 if it doesn't exist."""
    try:
        with open(filepath, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def _infer_model_from_filename(filename: str) -> Optional[str]:
    """Tries to guess the Odoo model from a CSV filename."""
    basename = Path(filename).stem
    # Remove common suffixes like _fail, _transformed, etc.
    clean_name = re.sub(r"(_fail|_transformed|_\d+)$", "", basename)
    # Convert underscores to dots
    model_name = clean_name.replace("_", ".")
    if "." in model_name:
        return model_name
    return None


def _get_fail_filename(model: str, is_fail_run: bool) -> str:
    """Generates a standardized filename for failed records.

    Args:
        model (str): The Odoo model name being imported.
        is_fail_run (bool): If True, indicates a recovery run, and a
            timestamp will be added to the filename.

    Returns:
        str: The generated filename for the fail file.
    """
    model_filename = model.replace(".", "_")
    if is_fail_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_filename}_{timestamp}_failed.csv"
    return f"{model_filename}_fail.csv"


def _run_preflight_checks(
    preflight_mode: PreflightMode, import_plan: dict[str, Any], **kwargs: Any
) -> bool:
    """Iterates through and runs all registered pre-flight checks.

    Args:
        preflight_mode (PreflightMode): The current mode (NORMAL or FAIL_MODE).
        import_plan (dict[str, Any]): A dictionary that checks can populate
            with strategy details (e.g., detected deferred fields).
        **kwargs (Any): A dictionary of arguments to pass to each check.

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    for check_func in preflight.PREFLIGHT_CHECKS:
        if not check_func(
            preflight_mode=preflight_mode, import_plan=import_plan, **kwargs
        ):
            return False
    return True


def run_import(  # noqa: C901
    config: Union[str, dict[str, Any]],
    filename: str,
    model: Optional[str],
    deferred_fields: Optional[list[str]],
    unique_id_field: Optional[str],
    no_preflight_checks: bool,
    headless: bool,
    worker: int,
    batch_size: int,
    skip: int,
    fail: bool,
    separator: str,
    ignore: Optional[list[str]],
    context: Any,  # Accept Any type for robustness
    encoding: str,
    o2m: bool,
    groupby: Optional[list[str]],
) -> None:
    """Main entry point for the import command, handling all orchestration."""
    log.info("Starting data import process from file...")

    parsed_context: dict[str, Any]
    if isinstance(context, str):
        try:
            loaded_context = json.loads(context)
            if not isinstance(loaded_context, dict):
                raise TypeError
            parsed_context = loaded_context
        except (json.JSONDecodeError, TypeError):
            _show_error_panel(
                "Invalid Context",
                "The --context argument must be a valid JSON dictionary string.",
            )
            return
    elif isinstance(context, dict):
        parsed_context = context
    else:
        _show_error_panel(
            "Invalid Context", "The context must be a dictionary or a JSON string."
        )
        return

    if not model:
        model = _infer_model_from_filename(filename)
        if not model:
            _show_error_panel(
                "Model Not Found",
                "Could not infer model from filename. Please use the --model option.",
            )
            return

    file_to_process = filename
    if fail:
        fail_path = Path(filename).parent / _get_fail_filename(model, False)
        line_count = _count_lines(str(fail_path))
        if line_count <= 1:
            Console().print(
                Panel(
                    f"No records to retry in '{fail_path}'.",
                    title="[bold green]No Recovery Needed[/bold green]",
                )
            )
            return
        log.info(
            f"Running in --fail mode. Retrying {line_count - 1} records from: "
            f"{fail_path}"
        )
        file_to_process = str(fail_path)
        if ignore is None:
            ignore = []
        if "_ERROR_REASON" not in ignore:
            log.info("Ignoring the internal '_ERROR_REASON' column for re-import.")
            ignore.append("_ERROR_REASON")

    import_plan: dict[str, Any] = {}
    if not no_preflight_checks:
        validation_filename = filename if fail else file_to_process
        if not _run_preflight_checks(
            preflight_mode=PreflightMode.FAIL_MODE if fail else PreflightMode.NORMAL,
            import_plan=import_plan,
            model=model,
            filename=file_to_process,
            validation_filename=validation_filename,
            config=config,
            headless=headless,
            separator=separator,
            unique_id_field=unique_id_field,
            ignore=ignore or [],
            o2m=o2m,
        ):
            return

    # --- Strategy Execution ---
    sorted_temp_file = None
    if import_plan.get("strategy") == "sort_and_one_pass_load":
        log.info("Executing 'Sort & One-Pass Load' strategy.")
        sorted_temp_file = sort.sort_for_self_referencing(
            file_to_process,
            id_column=import_plan["id_column"],
            parent_column=import_plan["parent_column"],
            encoding=encoding,
            separator=separator,
        )
        if isinstance(sorted_temp_file, str):
            file_to_process = sorted_temp_file
            # Disable deferred fields for this strategy
            deferred_fields = []

    final_deferred = deferred_fields or import_plan.get("deferred_fields", [])
    final_uid_field = unique_id_field or import_plan.get("unique_id_field") or "id"
    fail_output_file = str(Path(filename).parent / _get_fail_filename(model, fail))

    if fail:
        log.info("Single-record batching enabled for this import strategy.")
        max_conn = 1
        batch_size_run = 1
        force_create = True
    else:
        max_conn = worker
        batch_size_run = batch_size
        force_create = False

    start_time = time.time()
    try:
        # Use corrected file if preflight validation created one
        file_to_import = import_plan.get("_corrected_file", file_to_process)

        success, stats = import_threaded.import_data(
            config=config,
            model=model,
            unique_id_field=final_uid_field,
            file_csv=file_to_import,
            deferred_fields=final_deferred,
            context=parsed_context,
            fail_file=fail_output_file,
            encoding=encoding,
            separator=separator,
            ignore=ignore or [],
            max_connection=max_conn,
            batch_size=batch_size_run,
            skip=skip,
            force_create=force_create,
            o2m=o2m,
            split_by_cols=groupby,
        )
    finally:
        if (
            sorted_temp_file
            and sorted_temp_file is not True
            and os.path.exists(sorted_temp_file)
        ):
            os.remove(sorted_temp_file)

    elapsed = time.time() - start_time

    fail_file_was_created = _count_lines(fail_output_file) > 1
    is_truly_successful = success and not fail_file_was_created

    # Initialize id_map early to avoid UnboundLocalError
    id_map = (
        cast(dict[str, int], stats.get("id_map", {})) if is_truly_successful else {}
    )
    if is_truly_successful and id_map:
        if isinstance(config, str):
            cache.save_id_map(config, model, id_map)

        # --- Main Import Process ---
    log.info("*** STARTING MAIN IMPORT PROCESS ***")
    log.info(f"*** MODEL: {model} ***")
    log.info(f"*** FILENAME: {filename} ***")
    log.info(f"*** IMPORT PLAN KEYS: {list(import_plan.keys())} ***")
    if "strategies" in import_plan:
        log.info(f"*** IMPORT PLAN STRATEGIES: {import_plan['strategies']} ***")
        log.info(
            f"*** IMPORT PLAN STRATEGIES COUNT: {len(import_plan['strategies'])} ***"
        )
    else:
        log.info("*** NO STRATEGIES FOUND IN IMPORT PLAN ***")

    # --- Pass 1: Standard Fields ---
    if not fail:
        log.info("*** PASS 2: STARTING RELATIONAL IMPORT PROCESS ***")
        log.info(f"*** DETECTED STRATEGIES: {import_plan.get('strategies', {})} ***")
        log.info(f"*** STRATEGIES COUNT: {len(import_plan.get('strategies', {}))} ***")

        # Check if file exists and is not empty before reading
        if not os.path.exists(filename):
            log.warning(f"File does not exist: {filename}")
            return
        if os.path.getsize(filename) == 0:
            log.warning(f"File is empty: {filename}, skipping relational import")
            return

        # Read the CSV file with explicit schema for /id suffixed columns
        try:
            # First, get the header to determine if schema overrides are needed.
            header = pl.read_csv(
                filename, separator=separator, n_rows=0, truncate_ragged_lines=True
            ).columns
            id_columns = [col for col in header if col.endswith("/id")]
            schema_overrides = (
                {col: pl.Utf8 for col in id_columns} if id_columns else None
            )

            # Now, read the full file once with the correct schema and
            # encoding fallbacks.
            polars_encoding = _map_encoding_to_polars(encoding)
            try:
                source_df = pl.read_csv(
                    filename,
                    separator=separator,
                    encoding=polars_encoding,
                    truncate_ragged_lines=True,
                    schema_overrides=schema_overrides,
                )
            except (pl.exceptions.ComputeError, ValueError) as e:
                error_msg = str(e).lower()

                # Determine if this is an encoding error or a data type parsing error
                is_encoding_error = "encoding" in error_msg
                is_parse_error = "could not parse" in error_msg or "dtype" in error_msg

                if not is_encoding_error and not is_parse_error:
                    raise  # Not an encoding or parsing error, re-raise.

                if is_encoding_error:
                    # Handle encoding errors as before
                    log.warning(
                        f"Read failed with encoding '{encoding}', trying fallbacks..."
                    )
                    source_df = None
                    for enc in [
                        "utf8",
                        "windows-1252",
                        "latin-1",
                        "iso-8859-1",
                        "cp1252",
                    ]:
                        try:
                            source_df = pl.read_csv(
                                filename,
                                separator=separator,
                                encoding=_map_encoding_to_polars(enc),
                                truncate_ragged_lines=True,
                                schema_overrides=schema_overrides,
                            )
                            log.warning(
                                f"Successfully read with fallback encoding '{enc}'."
                            )
                            break
                        except (pl.exceptions.ComputeError, ValueError):
                            continue
                    if source_df is None:
                        raise ValueError(
                            "Could not read CSV with any of the tried encodings."
                        ) from e
                elif is_parse_error:
                    # This is a data type parsing error - try reading with
                    # flexible schema
                    log.warning(
                        f"Read failed due to data type parsing: '{e}'. "
                        f"Retrying with flexible parsing..."
                    )
                    try:
                        # Try reading with 'null_values' parameter and more
                        # flexible settings
                        source_df = pl.read_csv(
                            filename,
                            separator=separator,
                            encoding=polars_encoding,
                            truncate_ragged_lines=True,
                            schema_overrides=schema_overrides,
                            null_values=[
                                "",
                                "NULL",
                                "null",
                                "NaN",
                                "nan",
                            ],  # Handle common null representations
                        )
                        log.warning(
                            "Successfully read CSV with flexible parsing "
                            "for data type issues."
                        )
                    except (pl.exceptions.ComputeError, ValueError):
                        # If that still fails due to dtype issues, try with
                        # try_parse_dates=False
                        try:
                            source_df = pl.read_csv(
                                filename,
                                separator=separator,
                                encoding=polars_encoding,
                                truncate_ragged_lines=True,
                                schema_overrides=schema_overrides,
                                try_parse_dates=False,  # Don't try to auto-parse dates
                                null_values=["", "NULL", "null", "NaN", "nan"],
                            )
                            log.warning(
                                "Successfully read CSV by disabling date parsing."
                            )
                        except (pl.exceptions.ComputeError, ValueError):
                            # If still failing, read the data in a way that
                            # allows preflight to proceed
                            # The actual type validation and conversion will
                            # be handled during import
                            try:
                                # First get the header structure
                                header_info = pl.read_csv(
                                    filename,
                                    separator=separator,
                                    n_rows=0,
                                    truncate_ragged_lines=True,
                                ).columns

                                # Read with a limited number of rows to
                                # identify the issue
                                # and allow preflight to continue with basic
                                # data analysis
                                source_df = pl.read_csv(
                                    filename,
                                    separator=separator,
                                    encoding=polars_encoding,
                                    truncate_ragged_lines=True,
                                    schema_overrides={
                                        col: pl.Utf8 for col in header_info
                                    },  # All as strings for now
                                    n_rows=100,  # Only read first 100 rows
                                    # to ensure preflight performance
                                )
                                log.warning(
                                    "Successfully read partial CSV for "
                                    "preflight analysis. "
                                    "Type validation will be handled "
                                    "during actual import."
                                )
                            except (pl.exceptions.ComputeError, ValueError):
                                # Final attempt: read with maximum
                                # flexibility by skipping problematic rows
                                # Use ignore_errors to handle dtype parsing
                                # issues gracefully
                                source_df = pl.read_csv(
                                    filename,
                                    separator=separator,
                                    encoding=polars_encoding,
                                    truncate_ragged_lines=True,
                                    null_values=[
                                        "",
                                        "NULL",
                                        "null",
                                        "NaN",
                                        "nan",
                                        "N/A",
                                        "n/a",
                                    ],
                                    try_parse_dates=False,
                                    ignore_errors=True,
                                )
                                log.warning(
                                    "Successfully read CSV with error tolerance"
                                    " for preflight checks."
                                )
        except Exception as e:
            log.error(
                f"Failed to read source file '{filename}' for relational import: {e}"
            )
            return
        # At this point, source_df is guaranteed to be a DataFrame since
        # we would have returned early if there was an error.
        if source_df is None:
            # This should never happen due to the logic above, but as a safety check
            raise RuntimeError("source_df is unexpectedly None after CSV reading")

        # Only proceed with relational import if there are strategies defined
        strategies = import_plan.get("strategies", {})
        if strategies:
            with Progress() as progress:
                task_id = progress.add_task(
                    "Pass 2/2: Updating relations",
                    total=len(strategies),
                )
                for field, strategy_info in strategies.items():
                    log.info(
                        f"*** PROCESSING FIELD '{field}' WITH "
                        f"STRATEGY '{strategy_info['strategy']}' ***"
                    )
                    if strategy_info["strategy"] == "direct_relational_import":
                        log.info(
                            f"*** CALLING run_direct_relational_import "
                            f"for field '{field}' ***"
                        )
                        import_details = relational_import.run_direct_relational_import(
                            config,
                            model,
                            field,
                            strategy_info,
                            source_df,
                            id_map,
                            max_conn,
                            batch_size_run,
                            progress,
                            task_id,
                            filename,
                        )
                        if import_details:
                            log.info(
                                f"*** DIRECT RELATIONAL IMPORT RETURNED "
                                f"DETAILS FOR FIELD '{field}' ***"
                            )
                            import_threaded.import_data(
                                config=config,
                                model=import_details["model"],
                                unique_id_field=import_details["unique_id_field"],
                                file_csv=import_details["file_csv"],
                                max_connection=max_conn,
                                batch_size=batch_size_run,
                            )
                            Path(import_details["file_csv"]).unlink()
                        else:
                            log.info(
                                f"*** DIRECT RELATIONAL IMPORT RETURNED "
                                f"NONE FOR FIELD '{field}' ***"
                            )
                    elif strategy_info["strategy"] == "write_tuple":
                        log.info(
                            f"** CALLING run_write_tuple_import FOR FIELD '{field}' **"
                        )
                        result = relational_import.run_write_tuple_import(
                            config,
                            model,
                            field,
                            strategy_info,
                            source_df,
                            id_map,
                            max_conn,
                            batch_size_run,
                            progress,
                            task_id,
                            filename,
                        )
                        if not result:
                            log.warning(
                                f"Write tuple import failed for field '{field}'. "
                                "Check logs for details."
                            )
                    elif strategy_info["strategy"] == "write_o2m_tuple":
                        log.info(
                            f"*** CALLING run_write_o2m_tuple_import "
                            f"FOR FIELD '{field}' ***"
                        )
                        result = relational_import.run_write_o2m_tuple_import(
                            config,
                            model,
                            field,
                            strategy_info,
                            source_df,
                            id_map,
                            max_conn,
                            batch_size_run,
                            progress,
                            task_id,
                            filename,
                        )
                        if not result:
                            log.warning(
                                f"Write O2M tuple import failed for field '{field}'. "
                                "Check logs for details."
                            )

                    progress.update(task_id, advance=1)

        log.info(
            f"{stats.get('total_records', 0)} records processed. "
            f"Total time: {elapsed:.2f}s."
        )
        if final_deferred:  # It was a two-pass import
            summary = (
                f"Records: {stats.get('total_records', 0)}, "
                f"Created: {stats.get('created_records', 0)}, "
                f"Updated: {stats.get('updated_relations', 0)}"
            )
            title = f"[bold green]Import Complete for [cyan]{model}[/cyan][/bold green]"
            Console().print(
                Panel(
                    summary,
                    title=title,
                    expand=False,
                )
            )
        else:  # Single pass
            Console().print(
                Panel(
                    f"Import for [cyan]{model}[/cyan] finished successfully.",
                    title="[bold green]Import Complete[/bold green]",
                )
            )
    else:
        _show_error_panel(
            "Import Failed",
            "The import process failed. Check logs for details.",
        )


def run_import_for_migration(
    config: Union[str, dict[str, Any]],
    model: str,
    header: list[str],
    data: list[list[Any]],
    worker: int = 1,
    batch_size: int = 10,
) -> None:
    """Orchestrates the data import process from in-memory data.

    This function adapts in-memory data to the file-based import engine by
    writing the data to a temporary file. This allows it to leverage all the
    robust features of the main importer.

    Args:
        config (str): Path to the connection configuration file.
        model (str): The Odoo model to import data into.
        header (list[str]): A list of strings representing the column headers.
        data (list[list[Any]]): A list of lists representing the data rows.
        worker (int): The number of simultaneous connections to use.
        batch_size (int): The number of records to process in each batch.
    """
    log.info("Starting data import from in-memory data...")
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".csv", newline=""
        ) as tmp:
            writer = csv.writer(tmp)
            writer.writerow(header)
            writer.writerows(data)
            tmp_path = tmp.name
        log.info(f"In-memory data written to temporary file: {tmp_path}")
        import_threaded.import_data(
            config=config,
            model=model,
            unique_id_field="id",  # Migration import assumes 'id'
            file_csv=tmp_path,
            context={"tracking_disable": True},
            max_connection=int(worker),
            batch_size=int(batch_size),
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    log.info("In-memory import process finished.")
