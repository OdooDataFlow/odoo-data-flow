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


def write_file(
    filename: Optional[str] = None,
    header: Optional[list[str]] = None,
    data: Optional[list[list[Any]]] = None,
    fail: bool = False,
    model: str = "auto",
    launchfile: str = "import_auto.sh",
    worker: int = 1,
    batch_size: int = 10,
    init: bool = False,
    encoding: str = "utf-8",
    groupby: str = "",
    sep: str = ";",
    context: Optional[dict[str, Any]] = None,
    ignore: str = "",
    **kwargs: Any,  # to catch other unused params
) -> None:
    """Filewriter.

    Writes data to a CSV file and generates a corresponding shell script
    to import that file using the odoo-data-flow CLI.
    """
    # Step 1: Write the actual data file
    if filename and header is not None and data is not None:
        write_csv(filename, header, data, encoding=encoding)

    # Step 2: If no launchfile is specified, we are done.
    if not launchfile:
        return

    # Step 3: Only generate the import script if a filename was provided.
    if filename:
        # Determine the target model name
        if model == "auto":
            model_name = (
                os.path.basename(filename).replace(".csv", "").replace("_", ".")
            )
        else:
            model_name = model

        # Build the base command with its arguments
        # We use shlex.quote to ensure all arguments
        # are safely escaped for the shell.
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
            shlex.quote(encoding),
            "--worker",
            str(worker),
            "--size",
            str(batch_size),
            "--sep",
            shlex.quote(sep),
        ]

        # Add optional arguments if they have a value
        if groupby:
            command_parts.extend(["--groupby", shlex.quote(groupby)])
        if ignore:
            command_parts.extend(["--ignore", shlex.quote(ignore)])
        if context:
            command_parts.extend(["--context", shlex.quote(str(context))])

        # Write the command(s) to the shell script
        mode = "w" if init else "a"
        try:
            with open(launchfile, mode, encoding="utf-8") as f:
                # Write the main import command
                f.write(" ".join(command_parts) + "\n")

                # If fail mode is enabled,
                # write the second command with the --fail flag
                if fail:
                    fail_command_parts = [*command_parts, "--fail"]
                    f.write(" ".join(fail_command_parts) + "\n")
        except OSError as e:
            log.error(f"Failed to write to launch file {launchfile}: {e}")
