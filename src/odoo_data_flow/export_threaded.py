"""Export thread.

This module contains the low-level, multi-threaded logic for exporting
data from an Odoo instance.
"""

import csv
import sys
from time import time
from typing import Any, Optional

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
    ) -> None:
        """Initializes the export thread handler."""
        super().__init__(max_connection)
        self.model = model
        self.header = header
        self.context = context or {}
        self.results: dict[int, list[list[Any]]] = {}

    def launch_batch(self, data_ids: list[int], batch_number: int) -> None:
        """Submits a batch of IDs to be exported by a worker thread."""

        def launch_batch_fun(ids_to_export: list[int], num: int) -> None:
            start_time = time()
            try:
                log.debug(f"Exporting batch {num} with {len(ids_to_export)} records...")
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
        super().wait()

        all_data = []
        for batch_number in sorted(self.results.keys()):
            all_data.extend(self.results[batch_number])
        return all_data


def export_data(
    config_file: str,
    model: str,
    domain: list[Any],
    header: list[str],
    context: Optional[dict[str, Any]] = None,
    output: Optional[str] = None,
    max_connection: int = 1,
    batch_size: int = 100,
    separator: str = ";",
    encoding: str = "utf-8",
) -> tuple[Optional[list[str]], Optional[list[list[Any]]]]:
    """Export Data.

    The main function for exporting data. It can either write to a file or
    return the data in-memory for migrations.
    """
    try:
        connection = conf_lib.get_connection_from_config(config_file)
        model_obj = connection.get_model(model)
    except Exception as e:
        log.error(
            f"Failed to connect to Odoo or get model '{model}'. "
            f"Please check your configuration. Error: {e}"
        )
        return None, None

    rpc_thread = RPCThreadExport(max_connection, model_obj, header, context)
    start_time = time()

    log.info(f"Searching for records in model '{model}' to export...")
    ids = model_obj.search(domain, context=context)
    total_ids = len(ids)
    log.info(
        f"Found {total_ids} records to export. Splitting into batches of {batch_size}."
    )

    i = 0
    for id_batch in batch(ids, batch_size):
        rpc_thread.launch_batch(list(id_batch), i)
        i += 1

    all_exported_data = rpc_thread.get_data()

    log.info(
        f"Exported {len(all_exported_data)} records in total. Total time: "
        f"{time() - start_time:.2f}s."
    )

    if output:
        log.info(f"Writing exported data to file: {output}")
        try:
            with open(output, "w", newline="", encoding=encoding) as f:
                writer = csv.writer(f, delimiter=separator, quoting=csv.QUOTE_ALL)
                writer.writerow(header)
                writer.writerows(all_exported_data)
            log.info("File writing complete.")
        except OSError as e:
            log.error(f"Failed to write to output file {output}: {e}")
        return None, None
    else:
        log.info("Returning exported data in-memory.")
        return header, all_exported_data
