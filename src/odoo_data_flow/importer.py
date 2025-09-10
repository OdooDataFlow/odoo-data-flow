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
            parsed_context = json.loads(context)
            if not isinstance(parsed_context, dict):
                raise TypeError
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
        success, stats = import_threaded.import_data(
            config=config,
            model=model,
            unique_id_field=final_uid_field,
            file_csv=file_to_process,
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

    if is_truly_successful:
        id_map = cast(dict[str, int], stats.get("id_map", {}))
        if id_map:
            if isinstance(config, str):
                cache.save_id_map(config, model, id_map)

        # --- Pass 2: Relational Strategies ---
        if import_plan.get("strategies"):
            source_df = pl.read_csv(
                filename, separator=separator, truncate_ragged_lines=True
            )
            with Progress() as progress:
                task_id = progress.add_task(
                    "Pass 2/2: Relational fields",
                    total=len(import_plan["strategies"]),
                )
                for field, strategy_info in import_plan["strategies"].items():
                    if strategy_info["strategy"] == "direct_relational_import":
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
                            import_threaded.import_data(
                                config=config,
                                model=import_details["model"],
                                unique_id_field=import_details["unique_id_field"],
                                file_csv=import_details["file_csv"],
                                max_connection=max_conn,
                                batch_size=batch_size_run,
                            )
                            Path(import_details["file_csv"]).unlink()
                    elif strategy_info["strategy"] == "write_tuple":
                        relational_import.run_write_tuple_import(
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
                    elif strategy_info["strategy"] == "write_o2m_tuple":
                        relational_import.run_write_o2m_tuple_import(
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
