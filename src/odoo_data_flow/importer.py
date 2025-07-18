"""This module contains the core logic for importing data into Odoo."""

import ast
import os
from datetime import datetime
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel

from . import import_threaded
from .lib import preflight
from .lib.internal.ui import _show_error_panel
from .logging_config import log


def _run_preflight_checks(**kwargs: Any) -> bool:
    """Iterates through and runs all registered pre-flight checks."""
    for check_func in preflight.PREFLIGHT_CHECKS:
        if not check_func(**kwargs):
            return False
    return True


def run_import(
    config: str,
    filename: str,
    model: Optional[str] = None,
    no_preflight_checks: bool = False,
    headless: bool = False,
    worker: int = 1,
    batch_size: int = 10,
    skip: int = 0,
    fail: bool = False,
    separator: str = ";",
    split: Optional[str] = None,
    ignore: Optional[str] = None,
    check: bool = False,
    context: str = "{'tracking_disable' : True}",
    o2m: bool = False,
    encoding: str = "utf-8",
) -> None:
    """Orchestrates the data import process from a CSV file.

    Args:
        config: Path to the connection configuration file.
        filename: Path to the source CSV file to import.
        model: The Odoo model to import data into. If not provided, it's inferred
               from the filename.
        no_preflight_checks: If True, skips all pre-flight validation checks.
        headless: If True, runs in non-interactive mode, auto-confirming any
                  prompts (e.g., installing languages).
        worker: The number of simultaneous connections to use.
        batch_size: The number of records to process in each batch.
        skip: The number of initial lines to skip in the source file.
        fail: If True, runs in fail mode, retrying records from the _fail.csv file.
        separator: The delimiter used in the CSV file.
        split: The column name to group records by to avoid concurrent updates.
        ignore: A comma-separated string of column names to ignore.
        check: If True, checks if records were successfully imported.
        context: A string representation of the Odoo context dictionary.
        o2m: If True, enables special handling for one-to-many imports.
        encoding: The file encoding of the source file.
    """
    log.info("Starting data import process from file...")

    final_model = model
    if not final_model:
        base_name = os.path.basename(filename)
        inferred_model = os.path.splitext(base_name)[0].replace("_", ".")
        if not inferred_model or inferred_model.startswith("."):
            _show_error_panel(
                "Model Not Found",
                "Model not specified and could not be inferred from filename "
                f"'{base_name}'.\nPlease use the --model option.",
            )
            return
        final_model = inferred_model
        log.info(f"No model provided. Inferred model '{final_model}' from filename.")

    # --- Pre-flight Checks ---
    if not no_preflight_checks:
        if not _run_preflight_checks(
            model=final_model,
            filename=filename,
            config=config,
            headless=headless,
            separator=separator,
        ):
            return

    try:
        parsed_context = ast.literal_eval(context)
        if not isinstance(parsed_context, dict):
            raise TypeError("Context must be a dictionary.")
    except Exception as e:
        _show_error_panel(
            "Invalid Context",
            f"The --context argument must be a valid Python dictionary string.\n"
            f"Error: {e}",
        )
        return

    ignore_list = ignore.split(",") if ignore else []

    file_dir = os.path.dirname(filename)
    file_to_process: str
    fail_output_file: str
    is_fail_run: bool
    batch_size_run: int
    max_connection_run: int

    model_filename_part = final_model.replace(".", "_")

    if fail:
        log.info("Running in --fail mode. Retrying failed records...")
        file_to_process = os.path.join(file_dir, f"{model_filename_part}_fail.csv")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fail_output_file = os.path.join(
            file_dir, f"{model_filename_part}_{timestamp}_failed.csv"
        )
        batch_size_run = 1
        max_connection_run = 1
        is_fail_run = True
    else:
        file_to_process = filename
        fail_output_file = os.path.join(file_dir, f"{model_filename_part}_fail.csv")
        batch_size_run = int(batch_size)
        max_connection_run = int(worker)
        is_fail_run = False

    log.info(f"Importing file: {file_to_process}")
    log.info(f"Target model: {final_model}")
    log.info(f"Workers: {max_connection_run}, Batch Size: {batch_size_run}")
    log.info(f"Failed records will be saved to: {fail_output_file}")

    success = import_threaded.import_data(
        config,
        final_model,
        file_csv=file_to_process,
        context=parsed_context,
        fail_file=fail_output_file,
        encoding=encoding,
        separator=separator,
        ignore=ignore_list,
        split=split,
        check=check,
        max_connection=max_connection_run,
        batch_size=batch_size_run,
        skip=int(skip),
        o2m=o2m,
        is_fail_run=is_fail_run,
    )

    console = Console()
    if success:
        console.print(
            Panel(
                f"Import process for model [bold cyan]{final_model}[/bold cyan] "
                f"finished successfully.",
                title="[bold green]Import Complete[/bold green]",
                border_style="green",
            )
        )
    else:
        # log.error(
        _show_error_panel(
            "Import Aborted",
            "The import process was aborted due to a critical error. "
            "Please check the logs above for details.",
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

    parsed_context = {"tracking_disable": True}

    log.info(f"Importing {len(data)} records into model: {model}")
    log.info(f"Workers: {worker}, Batch Size: {batch_size}")

    import_threaded.import_data(
        config,
        model,
        header=header,
        data=data,
        context=parsed_context,
        max_connection=int(worker),
        batch_size=int(batch_size),
    )

    log.info("In-memory import process finished.")
