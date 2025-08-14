"""This module contains the high-level logic for exporting data from Odoo."""

import ast
from typing import Any, Optional

import polars as pl
from rich.console import Console
from rich.panel import Panel

from . import export_threaded
from .logging_config import log


def _show_error_panel(title: str, message: str) -> None:
    """Displays a formatted error panel to the console."""
    console = Console(stderr=True, style="bold red")
    console.print(Panel(message, title=title, border_style="red"))


def _show_success_panel(message: str) -> None:
    """Displays a formatted success panel to the console."""
    console = Console()
    console.print(
        Panel(
            message,
            title="[bold green]Export Complete[/bold green]",
            border_style="green",
        )
    )


def run_export(
    config: str,
    model: str,
    fields: str,
    output: str,
    domain: str = "[]",
    worker: int = 1,
    batch_size: int = 1000,
    context: str = "{}",
    separator: str = ";",
    encoding: str = "utf-8",
    technical_names: bool = False,
    streaming: bool = False,
    resume_session: Optional[str] = None,
) -> None:
    """Orchestrates the data export process."""
    log.info(f"Starting export for model '{model}'...")

    try:
        parsed_domain = ast.literal_eval(domain)
    except (ValueError, SyntaxError):
        _show_error_panel(
            "Invalid Domain",
            f"The provided domain string is not a valid Python literal: {domain}",
        )
        return

    try:
        parsed_context = ast.literal_eval(context)
        if not isinstance(parsed_context, dict):
            raise TypeError("Context must be a dictionary.")
    except Exception:
        _show_error_panel(
            "Invalid Context",
            f"The --context argument must be a valid Python dictionary string: "
            f"{context}",
        )
        return

    fields_list = fields.split(",")

    success, session_id, record_count, _ = export_threaded.export_data(
        config_file=config,
        model=model,
        domain=parsed_domain,
        header=fields_list,
        context=parsed_context,
        output=output,
        max_connection=int(worker),
        batch_size=int(batch_size),
        encoding=encoding,
        separator=separator,
        technical_names=technical_names,
        streaming=streaming,
        resume_session=resume_session,
    )

    if success:
        base_message = (
            f"Successfully streamed {record_count} records to "
            f"[bold cyan]{output}[/bold cyan]."
            if streaming
            else f"Successfully exported {record_count} records to "
            f"[bold cyan]{output}[/bold cyan]."
        )

        # --- Record Count Validation ---
        if output:
            try:
                actual_count = len(pl.read_csv(output, separator=separator))
                if actual_count == record_count:
                    final_message = (
                        f"{base_message}\n[green]Record count verified.[/green]"
                    )
                    _show_success_panel(final_message)
                else:
                    warning_message = (
                        f"{base_message}\n\n"
                        "[bold yellow]Warning:[/bold yellow] "
                        "Record count mismatch detected.\n"
                        f" - Expected: {record_count} records\n"
                        f" - Found:    {actual_count} records in the output file."
                    )
                    _show_error_panel("Count Validation Warning", warning_message)
            except Exception as e:
                log.warning(f"Could not validate record count in {output}: {e}")
                _show_success_panel(
                    base_message
                )  # Show original message if validation fails
        else:
            _show_success_panel(base_message)
    else:
        error_message = "The export process failed. Please check the logs for details."
        if session_id:
            error_message += (
                f"\n\nThis export was running under session ID: "
                f"[bold]{session_id}[/bold]"
                "\nTo resume this job, add the following flag to your command:"
                f"\n[bold cyan]--resume-session {session_id}[/bold cyan]"
            )
        _show_error_panel(
            "Export Failed",
            error_message,
        )


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

    success, _, _, result_df = export_threaded.export_data(
        config_file=config,
        model=model,
        domain=parsed_domain,
        header=fields,
        context=parsed_context,
        output=None,  # This signals the function to return data
        max_connection=int(worker),
        batch_size=int(batch_size),
        encoding=encoding,
        separator=";",
        technical_names=technical_names,
    )

    if not success or result_df is None:
        return fields, None

    header = result_df.columns
    # Corrected: Use a list comprehension to convert tuples to lists.
    data = [list(row) for row in result_df.iter_rows()]
    return header, data
