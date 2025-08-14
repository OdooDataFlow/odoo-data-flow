"""Write thread.

This module contains the low-level, multi-threaded logic for performing
batch 'write' operations on an Odoo instance.
"""

import concurrent.futures
import csv
import sys
from collections import defaultdict
from time import time
from typing import Any, Optional

import httpx
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)

from .lib import conf_lib
from .lib.internal.rpc_thread import RpcThread
from .lib.internal.tools import batch  # FIX: Add missing import
from .logging_config import log

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**30)


class RPCThreadWrite(RpcThread):
    """RPC Write Thread for handling batch updates."""

    def __init__(
        self,
        max_connection: int,
        model: Any,
        header: list[str],
        writer: Optional[Any] = None,
        context: Optional[dict[str, Any]] = None,
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
    ) -> None:
        """Initializes the write thread handler."""
        super().__init__(max_connection)
        self.model = model
        self.header = header
        self.writer = writer
        self.context = context or {}
        self.progress = progress
        self.task_id = task_id
        self.abort_flag = False

    def _execute_batch(self, lines: list[list[Any]], num: Any) -> dict[str, Any]:
        """Executes the write operation for a single batch of records."""
        if self.abort_flag:
            return {
                "processed": 0,
                "success": 0,
                "failed": 0,
                "error_summary": "Aborted",
            }

        start_time = time()
        summary: dict[str, Any] = {
            "processed": len(lines),
            "success": 0,
            "failed": 0,
        }
        error_summary = None

        try:
            id_index = self.header.index("id")
            grouped_updates: dict[frozenset[tuple[str, Any]], list[int]] = defaultdict(
                list
            )

            for row in lines:
                record_id = int(row[id_index])
                values_dict = {
                    self.header[i]: val
                    for i, val in enumerate(row)
                    if self.header[i] != "id"
                }
                dict_key = frozenset(values_dict.items())
                grouped_updates[dict_key].append(record_id)

            log.debug(
                f"Batch {num}: Grouped {len(lines)} updates into "
                f"{len(grouped_updates)} RPC calls."
            )

            for dict_items, record_ids in grouped_updates.items():
                values_to_write = dict(dict_items)
                try:
                    self.model.write(record_ids, values_to_write)
                    log.debug(
                        f"Successfully updated {len(record_ids)} "
                        f"records with: {values_to_write}"
                    )
                    summary["success"] += len(record_ids)
                except httpx.DecodingError:
                    error_summary = (
                        "Server returned invalid (non-JSON) response."
                        "Likely a proxy timeout."
                    )
                    log.error(f"Failed to process batch {num}. {error_summary}")
                    summary["failed"] += len(record_ids)
                except Exception as e:
                    error_summary = str(e)
                    log.error(f"Failed to update records {record_ids}: {error_summary}")
                    summary["failed"] += len(record_ids)
                    if self.writer:
                        for record_id in record_ids:
                            self.writer.writerow([record_id, error_summary])

        except Exception as e:
            error_summary = str(e)
            log.error(
                f"Batch {num} failed with an unexpected error: {e}",
                exc_info=True,
            )
            summary["failed"] = len(lines)

        log.info(
            f"Time for batch {num}: {time() - start_time:.2f}s. "
            f"Success: {summary['success']}, Failed: {summary['failed']}"
        )
        summary["error_summary"] = error_summary
        return summary

    def launch_batch(self, data_lines: list[list[Any]], batch_number: int) -> None:
        """Submits a batch of data lines to be written by a worker thread."""
        if self.abort_flag:
            return
        self.spawn_thread(self._execute_batch, [data_lines, batch_number])

    def wait(self) -> None:
        """Waits for tasks and updates the progress bar upon completion."""
        if not self.progress or self.task_id is None:
            super().wait()
            return

        shutdown_called = False
        for future in concurrent.futures.as_completed(self.futures):
            if self.abort_flag:
                if not shutdown_called:
                    self.executor.shutdown(wait=True, cancel_futures=True)
                    shutdown_called = True
                break
            try:
                result = future.result()
                if self.progress:
                    error_summary = result.get("error_summary")
                    if error_summary and len(error_summary) > 70:
                        error_summary = error_summary[:67] + "..."

                    self.progress.update(
                        self.task_id,
                        advance=result.get("processed", 0),
                        last_error=f"Last Error: {error_summary}"
                        if error_summary
                        else "",
                    )
            except Exception as e:
                log.error(f"A worker thread failed unexpectedly: {e}", exc_info=True)

        if not shutdown_called:
            self.executor.shutdown(wait=True)


def write_data(
    config_file: str,
    model: str,
    header: list[str],
    data: list[list[Any]],
    fail_file: str,
    max_connection: int = 1,
    batch_size: int = 1000,
    is_fail_run: bool = False,
    context: Optional[dict[str, Any]] = None,
    ignore: Optional[list[str]] = None,
    check: bool = False,
) -> bool:
    """Orchestrates the entire threaded write process.

    Args:
        config_file: Path to the connection configuration file.
        model: The Odoo model to write data to.
        header: A list of strings for the header row.
        data: A list of lists representing the data rows.
        fail_file: Path to write failed records to.
        max_connection: The number of simultaneous connections to use.
        batch_size: The number of records to process in each batch.
        context: A dictionary for the Odoo context.
        ignore: A list of column names to ignore during the write.
        check: If True, enables additional checks (currently a placeholder).
        is_fail_run: If True, indicates a run to re-process failed records.

    Returns:
        True if the write process completed without any failed records,
        False otherwise.
    """
    try:
        connection = conf_lib.get_connection_from_config(config_file)
        model_obj = connection.get_model(model)
    except Exception as e:
        log.error(f"Failed to connect to Odoo: {e}")
        return False

    fail_file_writer, fail_file_handle = None, None
    if fail_file:
        try:
            fail_file_handle = open(fail_file, "w", newline="", encoding="utf-8")
            fail_file_writer = csv.writer(
                fail_file_handle, delimiter=",", quoting=csv.QUOTE_ALL
            )
            fail_file_writer.writerow(["id", "_ERROR_REASON"])
        except OSError as e:
            log.error(f"Could not open fail file for writing: {fail_file}. Error: {e}")
            return False

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("•"),
        TextColumn("[green]{task.completed} of {task.total} records"),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("[bold red]{task.fields[last_error]}", justify="left"),
    )

    rpc_thread = None
    total_failed = 0
    try:
        with progress:
            task_id = progress.add_task(
                f"Writing to [bold]{model}[/bold]",
                total=len(data),
                last_error="",
            )
            rpc_thread = RPCThreadWrite(
                max_connection,
                model_obj,
                header,
                fail_file_writer,
                context,
                progress,
                task_id,
            )
            for i, lines_batch in enumerate(batch(data, batch_size)):
                rpc_thread.launch_batch(list(lines_batch), i)

            rpc_thread.wait()

    except KeyboardInterrupt:  # pragma: no cover
        log.warning("\nProcess interrupted by user. Shutting down workers...")
        if rpc_thread:
            rpc_thread.abort_flag = True
            rpc_thread.executor.shutdown(wait=True, cancel_futures=True)
        log.error("Write process aborted.")
        return False
    finally:
        if fail_file_handle:
            fail_file_handle.close()
        if rpc_thread:
            rpc_thread.executor.shutdown(wait=True)

    if rpc_thread and rpc_thread.futures:
        total_failed = sum(
            f.result().get("failed", 0)
            for f in rpc_thread.futures
            if f.done() and not f.cancelled()
        )

    return total_failed == 0
