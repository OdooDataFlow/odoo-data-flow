"""Import thread.

This module contains the low-level, multi-threaded logic for importing
data into an Odoo instance.
"""

import concurrent.futures
import csv
import sys
from collections.abc import Generator
from time import time
from typing import Any, Optional, TextIO

import requests  # type: ignore[import-untyped]
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
from .lib.internal.tools import batch
from .logging_config import log

# --- Fix for csv.field_size_limit OverflowError ---
# In newer Python versions (3.10+), especially on 64-bit systems,
# sys.maxsize is too large for the C long that the csv module's
# field_size_limit function expects. This causes an OverflowError.
# The following code block finds the maximum possible value that works
# by reducing it until it's accepted.
max_int = sys.maxsize
decrement = True
while decrement:
    decrement = False
    try:
        csv.field_size_limit(max_int)
    except OverflowError:
        max_int = int(max_int / 10)
        decrement = True


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
        writer: Optional[Any] = None,  # csv.writer is not a type, use Any
        context: Optional[dict[str, Any]] = None,
        add_error_reason: bool = False,
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
    ) -> None:
        """Initializes the import thread handler."""
        super().__init__(max_connection)
        self.model = model
        self.header = header
        self.writer = writer
        self.context = context or {}
        self.add_error_reason = add_error_reason
        self.progress = progress
        self.task_id = task_id
        self.abort_flag = False

    def _handle_odoo_messages(
        self, messages: list[dict[str, Any]], original_lines: list[list[Any]]
    ) -> list[list[Any]]:
        """Processes error messages from an Odoo load response.

        If a batch fails, ALL records from that batch are returned with an
        appropriate error message.
        """
        failed_lines_with_specific_errors = []
        failed_indices = set()
        full_error_message = ""

        # First, collect all records that Odoo reported with a specific error.
        for msg in messages:
            message = msg.get("message", "Unknown Odoo error")
            full_error_message += message + "\n"
            record_index = msg.get("record", -1)
            if 0 <= record_index < len(original_lines):
                # Make a copy to avoid modifying the original list
                failed_line = list(original_lines[record_index])
                if self.add_error_reason:
                    failed_line.append(message.replace("\n", " | "))
                failed_lines_with_specific_errors.append(failed_line)
                failed_indices.add(record_index)

        # If Odoo reports any error, the entire transaction is rolled back.
        # We must now account for the valid records that were also discarded.
        if failed_lines_with_specific_errors:
            collateral_error_msg = (
                "Record was valid but rolled back due to other errors in the batch."
            )
            for i, line in enumerate(original_lines):
                if i not in failed_indices:
                    # Make a copy to avoid modifying the original list
                    failed_line = list(line)
                    if self.add_error_reason:
                        failed_line.append(collateral_error_msg)
                    failed_lines_with_specific_errors.append(failed_line)
            return failed_lines_with_specific_errors

        # Fallback for generic batch errors without specific record details
        return self._handle_rpc_error(Exception(full_error_message), original_lines)

    def _handle_rpc_error(
        self, error: Exception, lines: list[list[Any]]
    ) -> list[list[Any]]:
        """Handles a general RPC exception, marking all lines as failed."""
        error_message = str(error).replace("\n", " | ")
        if self.add_error_reason:
            # Create a new list to avoid modifying the original lines list in place
            failed_lines = [list(line) for line in lines]
            for line in failed_lines:
                line.append(error_message)
            return failed_lines
        return lines

    def _handle_record_mismatch(
        self, response: dict[str, Any], lines: list[list[Any]]
    ) -> list[list[Any]]:
        """Handles the case where imported records don't match sent lines."""
        error_message = (
            f"Record count mismatch. Expected {len(lines)}, "
            f"got {len(response.get('ids', []))}. "
            "Probably a duplicate XML ID."
        )
        log.error(error_message)
        return self._handle_rpc_error(Exception(error_message), lines)

    def launch_batch(
        self,
        data_lines: list[list[Any]],
        batch_number: Any,
        check: bool = False,
    ) -> None:
        """Submits a batch of data lines to be imported by a worker thread."""
        if self.abort_flag:
            return

        def launch_batch_fun(lines: list[list[Any]], num: Any, do_check: bool) -> int:
            """The actual function executed by the worker thread."""
            if self.abort_flag:
                return 0
            start_time = time()
            failed_lines = []
            try:
                log.debug(f"Importing batch {num} with {len(lines)} records...")
                res = self.model.load(self.header, lines, context=self.context)

                if res.get("messages"):
                    failed_lines = self._handle_odoo_messages(res["messages"], lines)
                elif do_check and len(res.get("ids", [])) != len(lines):
                    failed_lines = self._handle_record_mismatch(res, lines)

            except requests.exceptions.ConnectionError as e:
                log.error(f"Connection to Odoo failed: {e}. Aborting import.")
                failed_lines = self._handle_rpc_error(e, lines)
                self.abort_flag = True
            except Exception as e:
                # For all other unexpected errors, log the full traceback.
                log.error(f"RPC call for batch {num} failed: {e}", exc_info=True)
                failed_lines = self._handle_rpc_error(e, lines)

            if failed_lines and self.writer:
                self.writer.writerows(failed_lines)

            success = not bool(failed_lines)
            log.info(
                f"Time for batch {num}: {time() - start_time:.2f}s. Success: {success}"
            )
            # Return the number of lines in this batch to update the progress
            return len(lines)

        self.spawn_thread(
            launch_batch_fun, [data_lines, batch_number], {"do_check": check}
        )

    def wait(self) -> None:
        """Waits for tasks and updates the progress bar upon completion."""
        if not self.progress or self.task_id is None:
            # Fallback to original behavior if no progress bar is provided
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
                # The number of processed lines is returned by the future
                num_processed = future.result()
                if self.progress:
                    self.progress.update(self.task_id, advance=num_processed)
            except Exception as e:
                log.error(f"A task in a worker thread failed: {e}", exc_info=True)

        if not shutdown_called:
            self.executor.shutdown(wait=True)


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
            reader = csv.reader(f, delimiter=separator)
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
    except ValueError as e:
        log.error(f"Failed to read file {file_path}: {e}")
        raise
    except Exception as e:
        log.error(f"Failed to read file {file_path}: {e}")
        return [], []


def _recursive_create_batches(
    current_data: list[list[Any]],
    group_cols: list[str],
    header: list[str],
    batch_size: int,
    o2m: bool,
    batch_prefix: str = "",
    level: int = 0,  # Current recursion level for internal tracking
) -> Generator[tuple[Any, list[list[Any]]], None, None]:
    """Recursively creates batches of data, optionally grouping by columns.

    This generator function processes data in batches. If `group_cols` are provided,
    it recursively groups records based on the values in these columns, ensuring
    that records belonging to the same group (and its subgroups) stay together
    within a batch. It also handles one-to-many (o2m) relationships, keeping
    child records with their parent records.

    Args:
        current_data (list[list[Any]]): The list of data rows to process. Each row
            is expected to be a list of values.
        group_cols (list[str]): A list of column names by which to group the data.
            The function processes these columns in order, from left to right.
        header (list[str]): The header row of the data, containing column names.
            This is used to find the index of `group_cols` and the 'id' column.
        batch_size (int): The maximum number of records allowed in a single batch
            before a new batch is started.
        o2m (bool): If True, enables one-to-many relationship handling. This means
            that records with an empty 'id' field (assuming the first column is 'id')
            will be kept with the preceding parent record in the same batch.
        batch_prefix (str, optional): A prefix string used to construct unique
            batch identifiers. This is primarily for internal tracking during recursion.
            Defaults to "".
        level (int, optional): The current recursion depth. Used internally for
            constructing unique batch identifiers. Defaults to 0.

    Yields:
        tuple[Any, list[list[Any]]]: A tuple where the first element is a unique
            identifier for the batch (string combining prefixes, levels, group
            counters, and group values) and the second element is the list of
            data rows constituting that batch.

    Raises:
        ValueError: If a column specified in `group_cols` is not found in the `header`.
            (This error is logged and the function returns in the current
            implementation, but a `ValueError` could conceptually be raised if
            not handled this way).
    """
    if not group_cols:
        # Base case: No more grouping columns, just yield regular batches
        for i, data_batch in enumerate(batch(current_data, batch_size)):
            yield (
                f"{batch_prefix}-{i}" if batch_prefix else str(i),
                list(data_batch),
            )
        return

    current_group_col = group_cols[0]
    remaining_group_cols = group_cols[1:]

    try:
        split_index = header.index(current_group_col)
        id_index = header.index("id")  # Needed for o2m logic
    except ValueError:
        # This error should ideally be caught earlier, as header is fixed
        log.error(
            f"Grouping column '{current_group_col}' not found in header. "
            "Cannot use --groupby."
        )
        return

    # Sort data based on the current grouping column
    # Empty strings/None/False values for the split column should come first
    # This also helps with consistent grouping
    # New (correct)
    current_data.sort(
        key=lambda row: (
            row[split_index] is None or row[split_index] == "",
            row[split_index],
        )
    )

    current_batch: list[list[Any]] = []
    current_split_value: Optional[str] = None
    group_counter = 0

    for row in current_data:
        row_split_value = row[split_index]
        # is_empty_value = row_split_value is None or row_split_value == ""
        is_o2m_line = (
            o2m and not row[id_index]
        )  # O2M lines should stay with their parent

        if not current_batch:
            # First row in a new segment, initialize current_split_value
            current_split_value = row_split_value
        elif not is_o2m_line and (
            row_split_value != current_split_value or len(current_batch) >= batch_size
        ):
            # If we've hit a new group value or max batch size,
            # process the current batch recursively
            yield from _recursive_create_batches(
                current_batch,
                remaining_group_cols,
                header,
                batch_size,
                o2m,
                f"{batch_prefix}{level}-{group_counter}-"
                f"{current_split_value or 'empty'}",
            )
            current_batch = []
            group_counter += 1
            current_split_value = row_split_value  # Start new group value

        current_batch.append(row)

    if current_batch:
        # Yield any remaining batch after the loop
        yield from _recursive_create_batches(
            current_batch,
            remaining_group_cols,
            header,
            batch_size,
            o2m,
            f"{batch_prefix}{level}-{group_counter}-{current_split_value or 'empty'}",
        )


def _create_batches(  # noqa C901
    data: list[list[Any]],
    split_by_cols: Optional[list[str]],
    header: list[str],
    batch_size: int,
    o2m: bool,
) -> Generator[tuple[int, list[list[Any]]], None, None]:
    """A generator that yields batches of data.

    If split_by_cols is provided, it
    groups records with the same value in that column into the same batch.
    """
    if not data:
        return

    batch: list[list[Any]] = []
    # Determine the grouping column. If multiple, we'll just use the first for now.
    # This might need refinement based on desired multi-column grouping behavior.
    split_column_name: Optional[str] = None
    if split_by_cols and len(split_by_cols) > 0:
        split_column_name = split_by_cols[0]

    split_index = -1
    if split_column_name:
        try:
            split_index = header.index(split_column_name)
        except ValueError:
            log.error(
                f"Grouping column '{split_column_name}' not found in header. "
                f"Cannot use --groupby."
            )
            return

    batch_counter = 0

    # If splitting is enabled and a valid column is found
    if split_index != -1 and split_column_name:
        current_data = sorted(
            data,
            key=lambda row: (
                row[split_index] is None or row[split_index] == "",
                row[split_index],
            ),
        )

        current_group_value = object()  # Sentinel for initial comparison

        for row in current_data:
            row_group_value = row[split_index]
            is_o2m_line = o2m and (
                row[0] is None or row[0] == ""
            )  # Assuming ID is first column

            if o2m and is_o2m_line and batch:
                # If it's an O2M line and a batch already exists,
                # append to the current batch
                batch.append(row)
            elif row_group_value != current_group_value:
                # New group or first record, yield current batch if not empty
                if batch:
                    batch_counter += 1
                    yield batch_counter, batch
                batch = [row]  # Start new batch with current row
                current_group_value = row_group_value
            elif len(batch) >= batch_size:
                # Same group, but batch is full, yield and start new batch
                batch_counter += 1
                yield batch_counter, batch
                batch = [row]
            else:
                # Same group, batch not full, append
                batch.append(row)

        if batch:  # Yield any remaining batch
            batch_counter += 1
            yield batch_counter, batch
    else:  # No splitting, just batch by size
        for row in data:
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch_counter, batch
                batch = []
        if batch:
            batch_counter += 1
            yield batch_counter, batch


def _setup_fail_file(
    fail_file: str,
    header: list[str],
    is_fail_run: bool,
    separator: str,
    encoding: str,
) -> tuple[Optional[Any], Optional[TextIO]]:
    """Opens the fail file and returns the writer and file handle."""
    try:
        fail_file_handle = open(fail_file, "w", newline="", encoding=encoding)
        fail_file_writer = csv.writer(
            fail_file_handle, delimiter=separator, quoting=csv.QUOTE_ALL
        )
        header_to_write = list(header)
        if is_fail_run:
            header_to_write.append("_ERROR_REASON")
        fail_file_writer.writerow(header_to_write)
        return fail_file_writer, fail_file_handle
    except OSError as e:
        log.error(f"Could not open fail file for writing: {fail_file}. Error: {e}")
        return None, None


def _execute_import_in_threads(
    max_connection: int,
    model_obj: Any,
    final_header: list[str],
    final_data: list[list[Any]],
    fail_file_writer: Optional[Any],
    context: dict[str, Any],
    is_fail_run: bool,
    split_by_cols: Optional[list[str]],
    batch_size: int,
    o2m: bool,
    check: bool,
    model: str,
) -> bool:
    """Sets up and runs the rich progress bar and threaded import."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("•"),
        TextColumn("[green]{task.completed} of {task.total} records"),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    rpc_thread = None
    try:
        with progress:
            task_id = progress.add_task(
                f"Importing to [bold]{model}[/bold]", total=len(final_data)
            )

            rpc_thread = RPCThreadImport(
                max_connection,
                model_obj,
                final_header,
                fail_file_writer,
                context,
                add_error_reason=is_fail_run,
                progress=progress,
                task_id=task_id,
            )

            for batch_number, lines_batch in _create_batches(
                final_data, split_by_cols, final_header, batch_size, o2m
            ):
                if rpc_thread.abort_flag:
                    log.error(
                        "Aborting further processing due to critical connection error."
                    )
                    break
                rpc_thread.launch_batch(lines_batch, batch_number, check)

            rpc_thread.wait()

    except KeyboardInterrupt:
        log.warning("\nImport process interrupted by user. Shutting down workers...")
        if rpc_thread:
            rpc_thread.abort_flag = True
            rpc_thread.executor.shutdown(wait=True, cancel_futures=True)
        log.error("Import aborted.")
        return False

    return not rpc_thread.abort_flag if rpc_thread else False


def import_data(
    config_file: str,
    model: str,
    header: Optional[list[str]] = None,
    data: Optional[list[list[Any]]] = None,
    file_csv: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    fail_file: Optional[str] = None,
    encoding: str = "utf-8",
    separator: str = ";",
    ignore: Optional[list[str]] = None,
    split_by_cols: Optional[list[str]] = None,
    check: bool = True,
    max_connection: int = 1,
    batch_size: int = 10,
    skip: int = 0,
    o2m: bool = False,
    is_fail_run: bool = False,
) -> bool:
    """Main function to orchestrate the import process.

    Can be run from a file or from in-memory data.

    Args:
        config_file: Path to the connection configuration file.
        model: The Odoo model to import data into.
        header: A list of strings for the header row (for in-memory data).
        data: A list of lists representing the data rows (for in-memory data).
        file_csv: Path to the source CSV file to import.
        context: A dictionary for the Odoo context.
        fail_file: Path to write failed records to.
        encoding: The file encoding of the source file.
        separator: The delimiter used in the CSV file.
        ignore: A list of column names to ignore during import.
        split_by_cols: The column names to group records by to avoid concurrent updates.
        check: If True, checks if records were successfully imported.
        max_connection: The number of simultaneous connections to use.
        batch_size: The number of records to process in each batch.
        skip: The number of initial lines to skip in the source file.
        o2m: If True, enables special handling for one-to-many imports.
        is_fail_run: If True, indicates a run to re-process failed records.

    Returns:
        True if the import completed, False if it was aborted.
    """
    _ignore = ignore or []
    _context = context or {}

    if file_csv:
        header, data = _read_data_file(file_csv, separator, encoding, skip)
        if not data and not header:
            return False

    if header is None or data is None:
        raise ValueError(
            "Please provide either a data file or both 'header' and 'data'."
        )

    # Filter out ignored columns from both header and data
    final_header, final_data = _filter_ignored_columns(_ignore, header, data)

    try:
        connection = conf_lib.get_connection_from_config(config_file)
        model_obj = connection.get_model(model)
    except Exception as e:
        log.error(f"Failed to connect to Odoo: {e}")
        return False

    fail_file_writer, fail_file_handle = None, None
    if fail_file:
        fail_file_writer, fail_file_handle = _setup_fail_file(
            fail_file, final_header, is_fail_run, separator, encoding
        )
        if not fail_file_writer:
            return False

    start_time = time()
    success = _execute_import_in_threads(
        max_connection,
        model_obj,
        final_header,
        final_data,
        fail_file_writer,
        _context,
        is_fail_run,
        split_by_cols,
        batch_size,
        o2m,
        check,
        model,
    )

    if fail_file_handle:
        fail_file_handle.close()
    log.info(
        f"{len(final_data)} records processed for model '{model}'. "
        f"Total time: {time() - start_time:.2f}s."
    )

    return success
