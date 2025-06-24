"""This module contains the core logic for importing data into Odoo."""

import ast

from . import import_threaded
from .logging_config import log


def run_import(
    config,
    filename,
    model,
    worker=1,
    batch_size=10,
    skip=0,
    fail=False,
    separator=";",
    split=None,
    ignore=None,
    check=False,
    context="{'tracking_disable' : True}",
    o2m=False,
    encoding="utf-8",
):
    """Orchestrates the data import process from a CSV file."""
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


def run_import_for_migration(config, model, header, data, worker=1, batch_size=10):
    """Orchestrates the data import process from in-memory data."""
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
