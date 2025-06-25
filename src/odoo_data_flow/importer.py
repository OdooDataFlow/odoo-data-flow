"""This module contains the core logic for importing data into Odoo."""

import ast
from typing import Any, Optional

from . import import_threaded
from .logging_config import log


def run_import(
    config: str,
    filename: str,
    model: str,
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
        model: The Odoo model to import data into.
        worker: The number of simultaneous connections to use.
        batch_size: The number of records to process in each batch.
        skip: The number of initial lines to skip in the source file.
        fail: If True, runs in fail mode, retrying records from the .fail file.
        separator: The delimiter used in the CSV file.
        split: The column name to group records by to avoid concurrent updates.
        ignore: A comma-separated string of column names to ignore.
        check: If True, checks if records were successfully imported.
        context: A string representation of the Odoo context dictionary.
        o2m: If True, enables special handling for one-to-many imports.
        encoding: The file encoding of the source file.
    """
    log.info("Starting data import process from file...")

    file_csv = filename
    fail_file = file_csv + ".fail"

    try:
        parsed_context = ast.literal_eval(context)
        if not isinstance(parsed_context, dict):
            raise TypeError("Context must be a dictionary.")
    except Exception as e:
        log.error(
            f"Invalid context provided. Must be a valid Python dictionary string. {e}"
        )
        return

    ignore_list = ignore.split(",") if ignore else []

    if fail:
        log.info("Running in --fail mode. Retrying failed records...")
        file_csv = fail_file
        fail_file = fail_file + ".bis"
        batch_size_run = 1
        max_connection_run = 1
    else:
        batch_size_run = int(batch_size)
        max_connection_run = int(worker)

    log.info(f"Importing file: {file_csv}")
    log.info(f"Target model: {model}")
    log.info(f"Workers: {max_connection_run}, Batch Size: {batch_size_run}")

    import_threaded.import_data(
        config,
        model,
        file_csv=file_csv,
        context=parsed_context,
        fail_file=fail_file,
        encoding=encoding,
        separator=separator,
        ignore=ignore_list,
        split=split,
        check=check,
        max_connection=max_connection_run,
        batch_size=batch_size_run,
        skip=int(skip),
        o2m=o2m,
    )

    log.info("Import process finished.")


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
