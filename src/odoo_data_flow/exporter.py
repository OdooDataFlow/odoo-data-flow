"""This module contains the core logic for exporting data from Odoo."""

import ast
from typing import Any, Optional

from . import export_threaded
from .logging_config import log


def run_export(
    config: str,
    filename: str,
    model: str,
    fields: str,
    domain: str = "[]",
    worker: int = 1,
    batch_size: int = 10,
    separator: str = ";",
    context: str = "{'tracking_disable' : True}",
    encoding: str = "utf-8",
) -> None:
    """Export runner.

    Orchestrates the data export process, writing the output to a CSV file.
    This function is designed to be called from the main CLI.
    """
    log.info("Starting data export process...")

    # Safely evaluate the domain and context strings
    try:
        parsed_domain = ast.literal_eval(domain)
        if not isinstance(parsed_domain, list):
            raise TypeError("Domain must be a list of tuples.")
    except Exception as e:
        log.error(f"Invalid domain provided. Must be a valid Python list string. {e}")
        return

    try:
        parsed_context = ast.literal_eval(context)
        if not isinstance(parsed_context, dict):
            raise TypeError("Context must be a dictionary.")
    except Exception as e:
        log.error(
            f"Invalid context provided. Must be a valid Python dictionary string. {e}"
        )
        return

    # Process the fields string into a list
    header = fields.split(",")

    log.info(f"Exporting from model: {model}")
    log.info(f"Output file: {filename}")
    log.info(f"Workers: {worker}, Batch Size: {batch_size}")

    # Call the core export function with an output filename
    export_threaded.export_data(
        config,
        model,
        parsed_domain,
        header,
        context=parsed_context,
        output=filename,
        max_connection=int(worker),
        batch_size=int(batch_size),
        separator=separator,
        encoding=encoding,
    )

    log.info("Export process finished.")


def run_export_for_migration(
    config: str,
    model: str,
    fields: list[str],
    domain: str = "[]",
    worker: int = 1,
    batch_size: int = 10,
    context: str = "{'tracking_disable' : True}",
    encoding: str = "utf-8",
) -> tuple[Optional[list[str]], Optional[list[list[Any]]]]:
    """Migration exporter.

    Orchestrates the data export process, returning the data in memory.
    This function is designed to be called by the migration tool.
    """
    log.info(f"Starting in-memory export from model '{model}' for migration...")

    try:
        parsed_domain = ast.literal_eval(domain)
    except Exception:
        log.warning(
            "Invalid domain string for migration export,"
            "defaulting to empty domain '[]'."
        )
        parsed_domain = []

    try:
        parsed_context = ast.literal_eval(context)
    except Exception:
        parsed_context = {}

    header, data = export_threaded.export_data(
        config,
        model,
        parsed_domain,
        fields,
        context=parsed_context,
        output=None,  # This signals the function to return data
        max_connection=int(worker),
        batch_size=int(batch_size),
        encoding=encoding,
        separator=";",  # Provide a default separator
    )

    if data:
        log.info(f"In-memory export complete. Fetched {len(data)} records.")
    else:
        log.info("In-memory export complete. No records fetched.")

    return header, data
