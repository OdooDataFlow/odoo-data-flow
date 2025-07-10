"""This module contains the core logic for exporting data from Odoo."""

import ast
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel

from . import export_threaded
from .logging_config import log


def _show_error_panel(title: str, message: str) -> None:
    """Displays a formatted error panel to the console."""
    console = Console(stderr=True, style="bold red")
    console.print(Panel(message, title=title, border_style="red"))


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
    technical_names: bool = False,
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
        _show_error_panel(
            "Invalid Domain",
            f"The --domain argument must be a valid Python list string.\nError: {e}",
        )
        return

    try:
        parsed_context = ast.literal_eval(context)
        if not isinstance(parsed_context, dict):
            raise TypeError("Context must be a dictionary.")
    except Exception as e:
        _show_error_panel(
            "Invalid Context",
            "The --context argument must be a valid Python dictionary string."
            f"\nError: {e}",
        )
        return

    # Process the fields string into a list
    header = fields.split(",")

    log.info(f"Exporting from model: {model}")
    log.info(f"Output file: {filename}")
    log.info(f"Workers: {worker}, Batch Size: {batch_size}")

    # Call the core export function with an output filename
    success, message = export_threaded.export_data_to_file(
        config,
        model,
        parsed_domain,
        header,
        output=filename,
        context=parsed_context,
        max_connection=int(worker),
        batch_size=int(batch_size),
        separator=separator,
        encoding=encoding,
        technical_names=technical_names,
    )

    console = Console()
    if success:
        console.print(
            Panel(
                f"Export process for model [bold cyan]{model}[/bold cyan] "
                f"finished successfully.",
                title="[bold green]Export Complete[/bold green]",
                border_style="green",
            )
        )
    else:
        _show_error_panel("Export Aborted", message)


def run_export_for_migration(
    config: str,
    model: str,
    fields: list[str],
    domain: str = "[]",
    worker: int = 1,
    batch_size: int = 10,
    context: str = "{'tracking_disable' : True}",
    encoding: str = "utf-8",
    technical_names: bool = False,
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

    header, data = export_threaded.export_data_for_migration(
        config,
        model,
        parsed_domain,
        fields,
        context=parsed_context,
        max_connection=int(worker),
        batch_size=int(batch_size),
        technical_names=technical_names,
    )

    if data:
        log.info(f"In-memory export complete. Fetched {len(data)} records.")
    else:
        log.info("In-memory export complete. No records fetched.")

    return header, data


def run_export_from_file(
    config: str,
    filename: str,
    worker: int = 1,
    batch_size: int = 10,
    separator: str = ";",
    context: str = "{'tracking_disable' : True}",
    encoding: str = "utf-8",
) -> None:
    """Export from file.

    This function is not yet implemented.
    It is intended to read export configurations from a file.
    """
    raise NotImplementedError("This feature is not implemented yet.")
