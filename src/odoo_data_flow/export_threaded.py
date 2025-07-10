"""Export thread.

This module contains the low-level, multi-threaded logic for exporting
data from an Odoo instance.
"""

import concurrent.futures
import csv
import sys
from time import time
from typing import Any, Optional

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .lib import conf_lib
from .lib.internal.rpc_thread import RpcThread
from .lib.internal.tools import batch
from .logging_config import log

# --- Fix for csv.field_size_limit OverflowError ---
max_int = sys.maxsize
decrement = True
while decrement:
    decrement = False
    try:
        csv.field_size_limit(max_int)
    except OverflowError:
        max_int = int(max_int / 10)
        decrement = True


class RPCThreadExport(RpcThread):
    """Export Thread handler.

    A specialized RpcThread for handling the export of data batches from Odoo.
    It collects results from multiple threads in a thread-safe manner.
    """

    def __init__(
        self,
        max_connection: int,
        model: Any,
        header: list[str],
        context: Optional[dict[str, Any]] = None,
        technical_names: bool = False,
    ) -> None:
        """Initializes the export thread handler."""
        super().__init__(max_connection)
        self.model = model
        self.header = header
        self.context = context or {}
        self.results: dict[int, list[list[Any]]] = {}
        self.technical_names = technical_names

    def launch_batch(self, data_ids: list[int], batch_number: int) -> None:
        """Submits a batch of IDs to be exported by a worker thread."""

        def launch_batch_fun(ids_to_export: list[int], num: int) -> None:
            start_time = time()
            try:
                log.debug(f"Exporting batch {num} with {len(ids_to_export)} records...")
                if self.technical_names:
                    records = self.model.read(ids_to_export, self.header)
                    datas = [
                        [record.get(field) for field in self.header]
                        for record in records
                    ]
                else:
                    datas = self.model.export_data(
                        ids_to_export, self.header, context=self.context
                    ).get("datas", [])
                self.results[num] = datas
                log.debug(
                    f"Batch {num} finished in {time() - start_time:.2f}s. "
                    f"Fetched {len(datas)} records."
                )
            except Exception as e:
                log.error(f"Export for batch {num} failed: {e}", exc_info=True)
                self.results[num] = []

        self.spawn_thread(launch_batch_fun, [data_ids, batch_number])

    def get_data(self) -> list[list[Any]]:
        """Get data.

        Waits for all threads to complete and returns the collected data
        in the correct order.
        """
        # The waiting is now handled by the progress bar in _fetch_export_data
        all_data = []
        for batch_number in sorted(self.results.keys()):
            all_data.extend(self.results[batch_number])
        return all_data


def _fetch_export_data(
    connection: Any,
    model_name: str,
    domain: list[Any],
    header: list[str],
    context: Optional[dict[str, Any]],
    max_connection: int,
    batch_size: int,
    technical_names: bool,
) -> list[list[Any]]:
    """Fetches data from Odoo using multithreading without writing to a file."""
    model_obj = connection.get_model(model_name)
    rpc_thread = RPCThreadExport(
        max_connection, model_obj, header, context, technical_names
    )
    start_time = time()

    log.info(f"Searching for records in model '{model_name}' to export...")
    ids = model_obj.search(domain, context=context)
    total_ids = len(ids)
    log.info(
        f"Found {total_ids} records to export. Splitting into batches of {batch_size}."
    )

    i = 0
    for id_batch in batch(ids, batch_size):
        rpc_thread.launch_batch(list(id_batch), i)
        i += 1

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("•"),
        TextColumn("[green]{task.completed} of {task.total} batches"),
        TextColumn("•"),
        TimeRemainingColumn(),
    )
    with progress:
        task = progress.add_task(
            f"[cyan]Exporting {model_name}...", total=len(rpc_thread.futures)
        )

        for future in concurrent.futures.as_completed(rpc_thread.futures):
            try:
                future.result()
            except Exception as e:
                log.error(f"A task in a worker thread failed: {e}", exc_info=True)
            finally:
                progress.update(task, advance=1)

    rpc_thread.executor.shutdown(wait=True)
    all_exported_data = rpc_thread.get_data()

    log.info(
        f"Exported {len(all_exported_data)} records in total. Total time: "
        f"{time() - start_time:.2f}s."
    )
    return all_exported_data


def export_data_to_file(
    config_file: str,
    model: str,
    domain: list[Any],
    header: list[str],
    output: str,
    context: Optional[dict[str, Any]] = None,
    max_connection: int = 1,
    batch_size: int = 100,
    separator: str = ";",
    encoding: str = "utf-8",
    technical_names: bool = False,
) -> tuple[bool, str]:
    """Export data to a file.

    Connects to Odoo, fetches data based on the domain, and writes it to a CSV file.
    """
    try:
        connection = conf_lib.get_connection_from_config(config_file)
    except Exception as e:
        message = (
            f"Failed to connect to Odoo. Please check your configuration. Error: {e}"
        )
        log.error(message)
        return False, message

    all_exported_data = _fetch_export_data(
        connection,
        model,
        domain,
        header,
        context,
        max_connection,
        batch_size,
        technical_names,
    )

    log.info(f"Writing exported data to file: {output}")
    try:
        with open(output, "w", newline="", encoding=encoding) as f:
            writer = csv.writer(f, delimiter=separator, quoting=csv.QUOTE_ALL)
            writer.writerow(header)
            writer.writerows(all_exported_data)
        log.info("File writing complete.")
        return True, "Export complete."
    except OSError as e:
        message = f"Failed to write to output file {output}: {e}"
        log.error(message)
        return False, message


def export_data_for_migration(
    config_file: str,
    model: str,
    domain: list[Any],
    header: list[str],
    context: Optional[dict[str, Any]] = None,
    max_connection: int = 1,
    batch_size: int = 100,
    technical_names: bool = False,
) -> tuple[list[str], Optional[list[list[Any]]]]:
    """Export data in-memory for migration.

    Connects to Odoo, fetches data, and returns it as a list of lists.
    """
    try:
        connection = conf_lib.get_connection_from_config(config_file)
    except Exception as e:
        message = (
            f"Failed to connect to Odoo. Please check your configuration. Error: {e}"
        )
        log.error(message)
        return header, None

    all_exported_data = _fetch_export_data(
        connection,
        model,
        domain,
        header,
        context,
        max_connection,
        batch_size,
        technical_names,
    )

    log.info("Returning exported data in-memory.")
    return header, all_exported_data
