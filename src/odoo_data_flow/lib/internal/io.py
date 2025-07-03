"""IO helpers.

This module contains low-level helper functions for file I/O,
including writing CSV data and generating shell scripts.
"""

import csv
import os
import shlex
from typing import Any, Optional

from ...logging_config import log


def write_csv(
    filename: str,
    header: list[str],
    data: list[list[Any]],
    encoding: str = "utf-8",
) -> None:
    """Writes data to a CSV file with a semicolon separator.

    Args:
        filename: The path to the output CSV file.
        header: A list of strings for the header row.
        data: A list of lists representing the data rows.
        encoding: The file encoding to use.
    """
    try:
        with open(filename, "w", newline="", encoding=encoding) as f:
            writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_ALL)
            writer.writerow(header)
            writer.writerows(data)
    except OSError as e:
        log.error(f"Failed to write to file {filename}: {e}")


def _build_import_command(
    filename: str, model: str, worker: int, batch_size: int, **kwargs: Any
) -> list[str]:
    """Builds the command parts for an 'import' shell command."""
    model_name = (
        os.path.basename(filename).replace(".csv", "").replace("_", ".")
        if model == "auto"
        else model
    )
    command_parts = [
        "odoo-data-flow",
        "import",
        "--config",
        shlex.quote(kwargs.get("conf_file", "conf/connection.conf")),
        "--file",
        shlex.quote(filename),
        "--model",
        shlex.quote(model_name),
        "--encoding",
        shlex.quote(kwargs.get("encoding", "utf-8")),
        "--worker",
        str(worker),
        "--size",
        str(batch_size),
        "--sep",
        shlex.quote(kwargs.get("sep", ";")),
    ]
    if kwargs.get("groupby"):
        command_parts.extend(["--groupby", shlex.quote(kwargs["groupby"])])
    if kwargs.get("ignore"):
        command_parts.extend(["--ignore", shlex.quote(kwargs["ignore"])])
    if kwargs.get("context"):
        command_parts.extend(["--context", shlex.quote(str(kwargs["context"]))])
    return command_parts


def _build_export_command(filename: str, model: str, **kwargs: Any) -> list[str]:
    """Builds the command parts for an 'export' shell command."""
    return [
        "odoo-data-flow",
        "export",
        "--config",
        shlex.quote(kwargs.get("conf_file", "conf/connection.conf")),
        "--file",
        shlex.quote(filename),
        "--model",
        shlex.quote(model),
        "--fields",
        shlex.quote(kwargs.get("fields", "")),
        "--domain",
        shlex.quote(kwargs.get("domain", "[]")),
        "--sep",
        shlex.quote(kwargs.get("sep", ";")),
        "--encoding",
        shlex.quote(kwargs.get("encoding", "utf-8")),
    ]


def write_file(
    filename: Optional[str] = None,
    header: Optional[list[str]] = None,
    data: Optional[list[list[Any]]] = None,
    fail: bool = False,
    model: str = "auto",
    launchfile: str = "import_auto.sh",
    command: str = "import",
    **kwargs: Any,
) -> None:
    """Writes data to a CSV and generates a corresponding shell script.

    This function can generate scripts for both `import` and `export` commands
    based on the `command` parameter.

    Args:
        filename: The path to the data file to be written or referenced.
        header: A list of strings for the header row.
        data: A list of lists representing the data rows.
        fail: If True (and command is 'import'), includes a second command
            with the --fail flag.
        model: The technical name of the Odoo model.
        launchfile: The path where the shell script will be saved.
        command: The command to generate in the script ('import' or 'export').
        **kwargs: Catches other command-specific params like 'worker', 'fields', etc.
    """
    if filename and header is not None and data is not None:
        write_csv(filename, header, data, encoding=kwargs.get("encoding", "utf-8"))

    if not launchfile or not filename:
        return

    command_parts: list[str]
    if command == "import":
        worker = kwargs.pop("worker", 1)
        batch_size = kwargs.pop("batch_size", 10)
        command_parts = _build_import_command(
            filename, model, worker, batch_size, **kwargs
        )
    elif command == "export":
        command_parts = _build_export_command(filename, model, **kwargs)
    else:
        log.error(f"Invalid command type '{command}' provided to write_file.")
        return

    mode = "w" if kwargs.get("init") else "a"
    try:
        with open(launchfile, mode, encoding="utf-8") as f:
            f.write(" ".join(command_parts) + "\n")
            if fail and command == "import":
                f.write(" ".join([*command_parts, "--fail"]) + "\n")
    except OSError as e:
        log.error(f"Failed to write to launch file {launchfile}: {e}")
