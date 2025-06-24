"""Import thread.

This module contains the low-level, multi-threaded logic for importing
data into an Odoo instance.
"""

import csv
import sys
from collections.abc import Generator
from time import time
from typing import Any, Optional

from ..logging_config import log
from .lib import conf_lib
from .lib.internal.rpc_thread import RpcThread
from .lib.internal.tools import batch

if sys.version_info.major >= 3:
    csv.field_size_limit(sys.maxsize)


class RPCThreadImport(RpcThread):
    """RPC Import Thread.

    A specialized RpcThread for handling the import of data batches into Odoo.
    It writes failed records to a file.
    """

    def __init__(
        self,
        max_connection: int,
        model: Any,
        header: list[str],
        writer: csv.writer,
        context: Optional[dict] = None,
    ):
        super().__init__(max_connection)
        self.model = model
        self.header = header
        self.writer = writer
        self.context = context or {}

    def launch_batch(
        self,
        data_lines: list[list[Any]],
        batch_number: Any,
        check: bool = False,
    ):
        """Submits a batch of data lines to be imported by a worker thread."""

        def launch_batch_fun(lines: list[list[Any]], num: Any, do_check: bool):
            start_time = time()
            success = False
            try:
                log.debug(f"Importing batch {num} with {len(lines)} records...")
                res = self.model.load(self.header, lines, context=self.context)

                if res.get("messages"):
                    for msg in res["messages"]:
                        record_index = msg.get("record", -1)
                        failed_line = (
                            lines[record_index] if record_index < len(lines) else "N/A"
                        )
                        log.error(
                            f"Odoo message for batch {num}: "
                            f"{msg.get('message', 'Unknown error')}. "
                            f"Record data: {failed_line}"
                        )
                    # Mark as failed if there are any error messages
                    success = False
                elif do_check and len(res.get("ids", [])) != len(lines):
                    log.error(
                        f"Record count mismatch for batch {num}. "
                        f"Expected {len(lines)}, "
                        f"got {len(res.get('ids', []))}. "
                        f"Probably a duplicate XML ID."
                    )
                    success = False
                else:
                    success = True

            except Exception as e:
                log.error(f"RPC call for batch {num} failed: {e}", exc_info=True)
                success = False

            if not success:
                self.writer.writerows(lines)

            log.info(
                f"Time for batch {num}: {time() - start_time:.2f}s. Success: {success}"
            )

        self.spawn_thread(
            launch_batch_fun, [data_lines, batch_number], {"check": check}
        )


def _filter_ignored_columns(
    ignore: list[str], header: list[str], data: list[list[Any]]
) -> tuple[list[str], list[list[Any]]]:
    """Removes ignored columns from header and data."""
    if not ignore:
        return header, data

    indices_to_keep = [i for i, h in enumerate(header) if h not in ignore]
    new_header = [header[i] for i in indices_to_keep]
    new_data = [[row[i] for i in indices_to_keep] for row in data]

    return new_header, new_data


def _read_data_file(
    file_path: str, separator: str, encoding: str, skip: int
) -> tuple[list[str], list[list[Any]]]:
    """Reads a CSV file and returns its header and data."""
    log.info(f"Reading data from file: {file_path}")
    try:
        with open(file_path, encoding=encoding, newline="") as f:
            reader = csv.reader(f, separator=separator)
            header = next(reader)

            if "id" not in header:
                raise ValueError(
                    "Source file must contain an 'id' column for external IDs."
                )

            if skip > 0:
                log.info(f"Skipping first {skip} lines...")
                for _ in range(skip):
                    next(reader)

            return header, [row for row in reader]
    except FileNotFoundError:
        log.error(f"Source file not found: {file_path}")
        return [], []
    except Exception as e:
        log.error(f"Failed to read file {file_path}: {e}")
        return [], []


def _create_batches(
    data: list[list[Any]],
    split_by_col: str,
    header: list[str],
    batch_size: int,
    o2m: bool,
) -> Generator[tuple[Any, list], None, None]:
    """A generator that yields batches of data.

    If split_by_col is provided, it
    groups records with the same value in that column into the same batch.
    """
    if not split_by_col:
        # Simple batching without grouping
        for i, data_batch in enumerate(batch(data, batch_size)):
            yield i, list(data_batch)
        return

    try:
        split_index = header.index(split_by_col)
        id_index = header.index("id")
    except ValueError as e:
        log.error(f"Grouping column '{e}' not found in header. Cannot use --groupby.")
        return

    # Sort data by the grouping column
    # to ensure all related records are contiguous
    data.sort(key=lambda row: row[split_index])

    current_batch = []
    current_split_value = None
    batch_num = 0

    for row in data:
        # For o2m, keep adding lines to the batch if the ID is empty
        is_o2m_line = o2m and not row[id_index]

        row_split_value = row[split_index]

        # Start a new batch if we are not in an o2m block and either the
        # split value changes or the batch size is reached.
        if (
            current_batch
            and not is_o2m_line
            and (
                row_split_value != current_split_value
                or len(current_batch) >= batch_size
            )
        ):
            yield f"{batch_num}-{current_split_value}", current_batch
            current_batch = []
            batch_num += 1

        current_batch.append(row)
        current_split_value = row_split_value

    if current_batch:
        yield f"{batch_num}-{current_split_value}", current_batch


def import_data(
    config_file: str,
    model: str,
    header: Optional[list[str]] = None,
    data: Optional[list[list[Any]]] = None,
    file_csv: Optional[str] = None,
    context: Optional[dict] = None,
    fail_file: str = False,
    encoding: str = "utf-8",
    separator: str = ";",
    ignore: Optional[list[str]] = None,
    split: str = False,
    check: bool = True,
    max_connection: int = 1,
    batch_size: int = 10,
    skip: int = 0,
    o2m: bool = False,
):
    """Main function to orchestrate the import process.

    Can be run from a file or from in-memory data.
    """
    ignore = ignore or []
    context = context or {}

    if file_csv:
        header, data = _read_data_file(file_csv, separator, encoding, skip)
        if not data:
            return  # Stop if file reading failed
        fail_file = fail_file or file_csv + ".fail"

    if header is None or data is None:
        raise ValueError(
            "Please provide either a data file or both 'header' and 'data'."
        )

    # Filter out ignored columns from both header and data
    header, data = _filter_ignored_columns(ignore, header, data)

    try:
        connection = conf_lib.get_connection_from_config(config_file)
        model_obj = connection.get_model(model)
    except Exception as e:
        log.error(f"Failed to connect to Odoo: {e}")
        return

    # Set up the writer for the fail file
    fail_file_writer = None
    fail_file_handle = None
    if fail_file:
        try:
            fail_file_handle = open(fail_file, "w", newline="", encoding=encoding)
            fail_file_writer = csv.writer(
                fail_file_handle, separator=separator, quoting=csv.QUOTE_ALL
            )
            fail_file_writer.writerow(header)  # Write header to fail file immediately
        except OSError as e:
            log.error(f"Could not open fail file for writing: {fail_file}. Error: {e}")
            return  # Cannot proceed without a fail file

    rpc_thread = RPCThreadImport(
        max_connection, model_obj, header, fail_file_writer, context
    )
    start_time = time()

    # Create batches and launch them in threads
    for batch_number, lines_batch in _create_batches(
        data, split, header, batch_size, o2m
    ):
        rpc_thread.launch_batch(lines_batch, batch_number, check)

    # Wait for all threads to complete
    rpc_thread.wait()

    if fail_file_handle:
        fail_file_handle.close()

    log.info(
        f"{len(data)} records processed for model '{model}'. "
        f"Total time: {time() - start_time:.2f}s."
    )
