"""Import thread.

This module contains the low-level, multi-threaded logic for importing
data into an Odoo instance.
"""

import ast
import concurrent.futures
import csv
import sys
import time  # noqa
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa
from typing import Any, Optional, TextIO, Union

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from .lib import conf_lib
from .lib.internal.rpc_thread import RpcThread
from .lib.internal.tools import batch
from .logging_config import log

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**30)


# --- Helper Functions ---
def _format_odoo_error(error: Any) -> str:
    """Tries to extract the meaningful message from an Odoo RPC error."""
    if not isinstance(error, str):
        error = str(error)
    try:
        error_dict = ast.literal_eval(error)
        if (
            isinstance(error_dict, dict)
            and "data" in error_dict
            and "message" in error_dict["data"]
        ):
            return str(error_dict["data"]["message"])
    except (ValueError, SyntaxError):
        pass
    return str(error).strip().replace("\n", " ")


def _read_data_file(
    file_path: str, separator: str, encoding: str, skip: int
) -> tuple[list[str], list[list[Any]]]:
    """Reads a CSV file and returns its header and data.

    This function handles opening and parsing a CSV file, skipping any
    initial lines as specified. It validates that an 'id' column exists,
    which is required for all import operations. It also handles common
    file I/O errors like FileNotFoundError.

    Args:
        file_path (str): The full path to the source CSV file.
        separator (str): The delimiter character used to separate columns.
        encoding (str): The character encoding of the file.
        skip (int): The number of lines to skip at the top of the file before
            reading the header.

    Returns:
        tuple[list[str], list[list[Any]]]: A tuple containing the header
        (as a list of strings) and the data (as a list of lists). Returns
        an empty tuple `([], [])` if the file cannot be read.

    Raises:
        ValueError: If the source file does not contain a required 'id' column.
    """
    try:
        with open(file_path, encoding=encoding, newline="") as f:
            reader = csv.reader(f, delimiter=separator)
            header = next(reader)
            if "id" not in header:
                raise ValueError("Source file must contain an 'id' column.")
            for _ in range(skip):
                next(reader)
            return header, list(reader)
    except FileNotFoundError:
        log.error(f"Source file not found: {file_path}")
        return [], []
    except Exception as e:
        log.error(f"Failed to read file {file_path}: {e}")
        if isinstance(e, ValueError):
            raise
        return [], []


def _filter_ignored_columns(
    ignore: list[str], header: list[str], data: list[list[Any]]
) -> tuple[list[str], list[list[Any]]]:
    """Removes ignored columns from header and data.

    This function filters a dataset by removing columns specified in the
    `ignore` list. It identifies the indices of columns to keep and rebuilds
    the header and each data row accordingly. If the `ignore` list is empty,
    it returns the original data and header without modification.

    Args:
        ignore (list[str]): A list of column header names to remove.
        header (list[str]): The original list of header columns.
        data (list[list[Any]]): The original data as a list of rows.

    Returns:
        tuple[list[str], list[list[Any]]]: A tuple containing two elements:
        the new header and the new data, both with the specified columns
        removed.
    """
    if not ignore:
        return header, data
    ignore_set = set(ignore)
    indices_to_keep = [
        i for i, h in enumerate(header) if h.split("/")[0] not in ignore_set
    ]
    new_header = [header[i] for i in indices_to_keep]

    if not indices_to_keep:
        return new_header, [[] for _ in data]

    max_index_needed = max(indices_to_keep)
    new_data = []
    for row_idx, row in enumerate(data):
        if len(row) <= max_index_needed:
            log.warning(
                f"Skipping malformed row {row_idx + 2}: has {len(row)} columns, "
                f"but header implies at least {max_index_needed + 1} are needed."
            )
            continue
        new_data.append([row[i] for i in indices_to_keep])

    return new_header, new_data


def _setup_fail_file(
    fail_file: Optional[str], header: list[str], separator: str, encoding: str
) -> tuple[Optional[Any], Optional[TextIO]]:
    """Opens the fail file and returns the writer and file handle."""
    if not fail_file:
        return None, None
    try:
        fail_handle = open(fail_file, "w", newline="", encoding=encoding)
        fail_writer = csv.writer(
            fail_handle, delimiter=separator, quoting=csv.QUOTE_ALL
        )
        header_to_write = list(header)
        if "_ERROR_REASON" not in header_to_write:
            header_to_write.append("_ERROR_REASON")
        fail_writer.writerow(header_to_write)
        return fail_writer, fail_handle
    except OSError as e:
        log.error(f"Could not open fail file for writing: {fail_file}. Error: {e}")
        return None, None


def _prepare_pass_2_data(
    all_data: list[list[Any]],
    header: list[str],
    unique_id_field_index: int,
    id_map: dict[str, int],
    deferred_fields: list[str],
) -> list[tuple[int, dict[str, Any]]]:
    """Prepares the list of write operations for Pass 2."""
    pass_2_data_to_write = []

    # FIX: Pre-calculate a map of deferred field names (e.g., 'parent_id')
    # to their actual index in the header.
    deferred_field_indices = {}
    deferred_fields_set = set(deferred_fields)
    for i, column_name in enumerate(header):
        field_base_name = column_name.split("/")[0]
        if field_base_name in deferred_fields_set:
            deferred_field_indices[field_base_name] = i

    for row in all_data:
        source_id = row[unique_id_field_index]
        db_id = id_map.get(source_id)
        if not db_id:
            continue

        update_vals = {}
        # Use the pre-calculated map to find the values to write.
        for field_name, field_index in deferred_field_indices.items():
            if field_index < len(row):
                related_source_id = row[field_index]
                if related_source_id:  # Ensure there is a value to look up
                    related_db_id = id_map.get(related_source_id)
                    if related_db_id:
                        update_vals[field_name] = related_db_id

        if update_vals:
            pass_2_data_to_write.append((db_id, update_vals))

    return pass_2_data_to_write  # This fixed it


def _recursive_create_batches(  # noqa: C901
    current_data: list[list[Any]],
    group_cols: list[str],
    header: list[str],
    batch_size: int,
    o2m: bool,
    batch_prefix: str = "",
    level: int = 0,
) -> Generator[tuple[Any, list[list[Any]]], None, None]:
    """Recursively creates batches of data, handling grouping and o2m."""
    if not group_cols:
        # Base case: No more grouping, handle o2m or simple batching
        current_batch: list[list[Any]] = []
        try:
            id_index = header.index("id")
        except ValueError:
            # If no 'id' column, o2m cannot work, so just batch by size
            for i, data_batch in enumerate(batch(current_data, batch_size)):
                yield (f"{batch_prefix}-{i}", list(data_batch))
            return

        for row in current_data:
            is_new_parent = o2m and row[id_index] and current_batch
            is_batch_full = not o2m and len(current_batch) >= batch_size

            if is_new_parent or is_batch_full:
                yield (current_batch[0][id_index], current_batch)
                current_batch = []

            current_batch.append(row)

        if current_batch:
            yield (current_batch[0][id_index], current_batch)
        return

    current_group_col, remaining_group_cols = group_cols[0], group_cols[1:]
    try:
        split_index = header.index(current_group_col)
    except ValueError:
        log.error(
            f"Grouping column '{current_group_col}' not found. Cannot use --groupby."
        )
        return

    current_data.sort(
        key=lambda r: (
            r[split_index] is None or r[split_index] == "",
            r[split_index],
        )
    )
    current_batch, current_split_value, group_counter = [], None, 0
    for row in current_data:
        row_split_value = row[split_index]
        if not current_batch:
            current_split_value = row_split_value
        elif row_split_value != current_split_value:
            yield from _recursive_create_batches(
                current_batch,
                remaining_group_cols,
                header,
                batch_size,
                o2m,
                f"{batch_prefix}{level}-{group_counter}-"
                f"{current_split_value or 'empty'}",
            )
            current_batch, group_counter, current_split_value = (
                [],
                group_counter + 1,
                row_split_value,
            )
        current_batch.append(row)

    if current_batch:
        yield from _recursive_create_batches(
            current_batch,
            remaining_group_cols,
            header,
            batch_size,
            o2m,
            f"{batch_prefix}{level}-{group_counter}-{current_split_value or 'empty'}",
        )


def _create_batches(
    data: list[list[Any]],
    split_by_cols: Optional[list[str]],
    header: list[str],
    batch_size: int,
    o2m: bool,
) -> Generator[tuple[int, list[list[Any]]], None, None]:
    """A generator that yields batches of data, starting the recursive batching."""
    if not data:
        return
    for i, (_, batch_data) in enumerate(
        _recursive_create_batches(data, split_by_cols or [], header, batch_size, o2m),
        start=1,
    ):
        yield i, batch_data


class RPCThreadImport(RpcThread):
    """A specialized RpcThread for handling data import and write tasks."""

    def __init__(
        self,
        max_connection: int,
        progress: Progress,
        task_id: TaskID,
        writer: Optional[Any] = None,
        fail_handle: Optional[TextIO] = None,
    ) -> None:
        super().__init__(max_connection)
        (
            self.progress,
            self.task_id,
            self.writer,
            self.fail_handle,
            self.abort_flag,
        ) = (
            progress,
            task_id,
            writer,
            fail_handle,
            False,
        )


def _convert_external_id_field(
    model: Any,
    field_name: str,
    field_value: str,
) -> tuple[str, Any]:
    """Convert an external ID field to a database ID.

    Args:
        model: The Odoo model object
        field_name: The field name (e.g., 'parent_id/id')
        field_value: The external ID value

    Returns:
        Tuple of (base_field_name, converted_value)
    """
    base_field_name = field_name[:-3]  # Remove '/id' suffix
    converted_value = False

    if not field_value:
        # Empty external ID means no value for this field
        log.debug(
            f"Converted empty external ID {field_name} -> {base_field_name} (False)"
        )
    else:
        # Convert external ID to database ID
        try:
            # Look up the database ID for this external ID
            record_ref = model.env.ref(field_value, raise_if_not_found=False)
            if record_ref:
                converted_value = record_ref.id
                log.debug(
                    f"Converted external ID {field_name} ({field_value}) -> "
                    f"{base_field_name} ({record_ref.id})"
                )
            else:
                # If we can't find the external ID, value remains False
                log.warning(
                    f"Could not find record for external ID '{field_value}', "
                    f"setting {base_field_name} to False"
                )
        except Exception as e:
            log.warning(
                f"Error looking up external ID '{field_value}' for field "
                f"'{field_name}': {e}"
            )
            # On error, value remains False

    return base_field_name, converted_value


def _process_external_id_fields(
    model: Any,
    clean_vals: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Process all external ID fields in the clean values.

    Args:
        model: The Odoo model object
        clean_vals: Dictionary of clean field values

    Returns:
        Tuple of (converted_vals, external_id_fields)
    """
    converted_vals: dict[str, Any] = {}
    external_id_fields: list[str] = []

    for field_name, field_value in clean_vals.items():
        # Handle external ID references (e.g., 'parent_id/id' -> 'parent_id')
        if field_name.endswith("/id"):
            # _convert_external_id_field is now a pure function that returns
            # (base_field_name, converted_value) instead of modifying converted_vals as side effect
            base_name, value = _convert_external_id_field(
                model, field_name, field_value
            )
            converted_vals[base_name] = value
            external_id_fields.append(field_name)
        else:
            # Regular field - pass through as-is
            converted_vals[field_name] = field_value

    return converted_vals, external_id_fields


def _handle_create_error(
    i: int,
    create_error: Exception,
    line: list[Any],
    error_summary: str,
) -> tuple[str, list[Any], str]:
    """Handle errors during record creation.

    Args:
        i: The row index
        create_error: The exception that occurred
        line: The data line being processed
        error_summary: Current error summary

    Returns:
        Tuple of (error_message, failed_line, error_summary)
    """
    error_str = str(create_error)
    error_str_lower = error_str.lower()

    if "tuple index out of range" in error_str_lower or "indexerror" in error_str_lower:
        error_message = f"Tuple unpacking error in row {i + 1}: {create_error}"
        if "Fell back to create" in error_summary:
            error_summary = "Tuple unpacking error detected"
    else:
        error_message = error_str.replace("\n", " | ")
        if "invalid field" in error_str_lower and "/id" in error_str_lower:
            error_message = (
                f"Invalid external ID field detected in row {i + 1}: {error_message}"
            )

        if "Fell back to create" in error_summary:
            error_summary = error_message

    failed_line = [*line, error_message]
    return error_message, failed_line, error_summary


def _create_batch_individually(
    model: Any,
    batch_lines: list[list[Any]],
    batch_header: list[str],
    uid_index: int,
    context: dict[str, Any],
    ignore_list: list[str],
) -> dict[str, Any]:
    """Fallback to create records one-by-one to get detailed errors."""
    id_map: dict[str, int] = {}
    failed_lines: list[list[Any]] = []
    error_summary = "Fell back to create"
    header_len = len(batch_header)
    ignore_set = set(ignore_list)

    for i, line in enumerate(batch_lines):
        try:
            if len(line) != header_len:
                raise IndexError(
                    f"Row has {len(line)} columns, but header has {header_len}."
                )

            source_id = line[uid_index]
            # 1. SEARCH BEFORE CREATE
            existing_record = model.browse().env.ref(
                f"__export__.{source_id}", raise_if_not_found=False
            )

            if existing_record:
                id_map[source_id] = existing_record.id
                continue

            # 2. PREPARE FOR CREATE
            vals = dict(zip(batch_header, line))
            clean_vals = {
                k: v
                for k, v in vals.items()
                if k.split("/")[0] not in ignore_set
                # Allow external ID fields through for conversion
            }

            # 3. CREATE
            # Convert external ID references to actual database IDs before creating
            converted_vals, external_id_fields = _process_external_id_fields(
                model, clean_vals
            )

            log.debug(f"External ID fields found: {external_id_fields}")
            log.debug(f"Converted vals keys: {list(converted_vals.keys())}")

            new_record = model.create(converted_vals, context=context)
            id_map[source_id] = new_record.id
        except IndexError as e:
            error_message = f"Malformed row detected (row {i + 1} in batch): {e}"
            failed_lines.append([*line, error_message])
            if "Fell back to create" in error_summary:
                error_summary = "Malformed CSV row detected"
            continue
        except Exception as create_error:
            error_message, new_failed_line, error_summary = _handle_create_error(
                i, create_error, line, error_summary
            )
            failed_lines.append(new_failed_line)
            if "Fell back to create" in error_summary:
                error_summary = error_message
    return {
        "id_map": id_map,
        "failed_lines": failed_lines,
        "error_summary": error_summary,
    }


def _execute_load_batch(
    thread_state: dict[str, Any],
    batch_lines: list[list[Any]],
    batch_header: list[str],
    batch_number: int,
) -> dict[str, Any]:
    """Executes a batch import with dynamic scaling and `create` fallback.

    This is the core worker for Pass 1. It processes a given batch of records
    by first attempting a fast `load`. If a memory or gateway-related error
    (like a 502) is detected, it automatically reduces the size of the data
    chunks it sends and retries. For other errors, it falls back to a
    record-by-record `create` for only the failed chunk.

    Args:
        thread_state (dict[str, Any]): Shared state from the orchestrator.
        batch_lines (list[list[Any]]): The list of data rows for this batch.
        batch_header (list[str]): The list of header columns for this batch.
        batch_number (int): The identifier for this batch, used for logging.

    Returns:
        dict[str, Any]: A dictionary containing the aggregated results for
        the entire batch, including `id_map` and `failed_lines`.
    """
    model, context, progress = (
        thread_state["model"],
        thread_state.get("context", {"tracking_disable": True}),
        thread_state["progress"],
    )
    uid_index = thread_state["unique_id_field_index"]
    ignore_list = thread_state.get("ignore_list", [])

    if thread_state.get("force_create"):
        progress.console.print(
            f"Batch {batch_number}: Fail mode active, using `create` method."
        )
        result = _create_batch_individually(
            model, batch_lines, batch_header, uid_index, context, ignore_list
        )
        result["success"] = bool(result.get("id_map"))
        return result

    lines_to_process = list(batch_lines)
    aggregated_id_map: dict[str, int] = {}
    aggregated_failed_lines: list[list[Any]] = []
    chunk_size = len(lines_to_process)

    while lines_to_process:
        current_chunk = lines_to_process[:chunk_size]
        load_header, load_lines = batch_header, current_chunk

        if ignore_list:
            ignore_set = set(ignore_list)
            indices_to_keep = [
                i
                for i, h in enumerate(batch_header)
                if h.split("/")[0] not in ignore_set
            ]
            load_header = [batch_header[i] for i in indices_to_keep]
            max_index = max(indices_to_keep) if indices_to_keep else 0
            load_lines = [
                [row[i] for i in indices_to_keep]
                for row in current_chunk
                if len(row) > max_index
            ]

        if not load_lines:
            lines_to_process = lines_to_process[chunk_size:]
            continue

        try:
            log.debug(f"Attempting `load` for chunk of batch {batch_number}...")
            res = model.load(load_header, load_lines, context=context)
            if res.get("messages"):
                error = res["messages"][0].get("message", "Batch load failed.")
                raise ValueError(error)

            created_ids = res.get("ids", [])
            if len(created_ids) != len(load_lines):
                raise ValueError("Record count mismatch after load.")

            id_map = {
                line[uid_index]: created_ids[i] for i, line in enumerate(current_chunk)
            }
            aggregated_id_map.update(id_map)
            lines_to_process = lines_to_process[chunk_size:]

        except Exception as e:
            error_str = str(e).lower()

            # SPECIAL CASE: Client-side timeouts for local processing
            # These should be IGNORED entirely to allow long server processing
            if (
                "timed out" == error_str.strip()
                or "read timeout" in error_str
                or type(e).__name__ == "ReadTimeout"
            ):
                log.debug(f"Client-side timeout detected ({type(e).__name__}): {e}")
                log.debug(
                    "Ignoring client-side timeout to allow server processing"
                    " to continue"
                )
                # CRITICAL: For local imports, ignore client timeouts completely
                # This restores the previous behavior where long processing was allowed
                progress.console.print(
                    f"[yellow]INFO:[/] Batch {batch_number} processing on server. "
                    f"Continuing to wait for completion..."
                )
                # Continue with next chunk WITHOUT fallback - let server finish
                lines_to_process = lines_to_process[chunk_size:]
                continue

            # For all other exceptions, use the original scalable error detection
            is_scalable_error = (
                "memory" in error_str
                or "out of memory" in error_str
                or "502" in error_str
                or "gateway" in error_str
                or "proxy" in error_str
                or "timeout" in error_str
            )

            if is_scalable_error and chunk_size > 1:
                chunk_size = max(1, chunk_size // 2)
                progress.console.print(
                    f"[yellow]WARN:[/] Batch {batch_number} hit scalable error. "
                    f"Reducing chunk size to {chunk_size} and retrying."
                )
                continue

            clean_error = str(e).strip().replace("\n", " ")
            progress.console.print(
                f"[yellow]WARN:[/] Batch {batch_number} failed `load` "
                f"('{clean_error}'). "
                f"Falling back to `create` for {len(current_chunk)} records."
            )
            fallback_result = _create_batch_individually(
                model,
                current_chunk,
                batch_header,
                uid_index,
                context,
                ignore_list,
            )
            aggregated_id_map.update(fallback_result.get("id_map", {}))
            aggregated_failed_lines.extend(fallback_result.get("failed_lines", []))
            lines_to_process = lines_to_process[chunk_size:]

    return {
        "id_map": aggregated_id_map,
        "failed_lines": aggregated_failed_lines,
        "success": True,
    }


def _execute_write_batch(
    thread_state: dict[str, Any],
    batch_writes: tuple[list[int], dict[str, Any]],
    batch_number: int,
) -> dict[str, Any]:
    """Executes a batch of write operations for a group of records.

    This is the core worker function for Pass 2. It takes a list of database
    IDs and a single dictionary of values and updates all records in one RPC call.

    Args:
        thread_state (dict[str, Any]): Shared state from the orchestrator,
            containing the Odoo model object.
        batch_writes (tuple[list[int], dict[str, Any]]): A tuple containing
            the list of database IDs and the dictionary of values to write.
        batch_number (int): The identifier for this batch, used for logging.

    Returns:
        dict[str, Any]: A dictionary containing the results of the batch,
        with a `failed_writes` key if the operation failed.
    """
    model = thread_state["model"]
    context = thread_state.get("context", {})  # Get context
    ids, vals = batch_writes
    try:
        # The core of the fix: use model.write(ids, vals) for batch updates.
        model.write(ids, vals, context=context)
        return {
            "failed_writes": [],
            "successful_writes": len(ids),
            "success": True,
        }
    except Exception as e:
        error_message = str(e).replace("\n", " | ")
        # If the batch fails, all IDs in it are considered failed.
        failed_writes = [(db_id, vals, error_message) for db_id in ids]
        return {
            "failed_writes": failed_writes,
            "error_summary": error_message,
            "successful_writes": 0,
            "success": False,
        }


def _run_threaded_pass(  # noqa: C901
    rpc_thread: RPCThreadImport,
    target_func: Any,
    batches: Iterable[tuple[int, Any]],
    thread_state: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Orchestrates a multi-threaded pass and aggregates results.

    This is a generic function that manages a multi-threaded operation,
    used for both Pass 1 (load/create) and Pass 2 (write). It spawns worker
    threads for each batch of data and then collects and aggregates the
    results as they are completed, updating the progress bar in real-time.

    Args:
        rpc_thread (RPCThreadImport): The thread manager instance that controls
            the thread pool and progress bar.
        target_func (Any): The worker function to be executed in each thread
            (e.g., `_execute_load_batch`).
        batches (Iterable[tuple[int, Any]]): An iterable that yields
            batches of data, where each item is a tuple of `(batch_number,
            batch_data)`. The type of `batch_data` can vary between passes.
        thread_state (dict[str, Any]): A dictionary of shared state to be
            passed to each worker function.

    Returns:
        tuple[dict[str, Any], bool]: A typle and a dictionary containing
        the aggregated results from all
        worker threads, such as `id_map` and `failed_lines`.
    """
    # This logic is brittle but preserved to minimize unrelated changes.
    # It dynamically constructs arguments based on the target function name.
    futures = {
        rpc_thread.spawn_thread(
            target_func,
            [thread_state, data, num]
            if target_func.__name__ == "_execute_write_batch"
            else [thread_state, data, thread_state.get("batch_header"), num],
        )
        for num, data in batches
        if not rpc_thread.abort_flag
    }

    aggregated: dict[str, Any] = {
        "id_map": {},
        "failed_lines": [],
        "failed_writes": [],
        "successful_writes": 0,
    }
    consecutive_failures = 0
    successful_batches = 0
    original_description = rpc_thread.progress.tasks[rpc_thread.task_id].description

    try:
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                is_successful_batch = result.get("success", False)
                if is_successful_batch:
                    successful_batches += 1
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 50:
                        log.error(
                            f"Aborting import: Multiple "
                            f"({consecutive_failures}) consecutive batches have"
                            f" failed."
                        )
                        rpc_thread.abort_flag = True

                aggregated["id_map"].update(result.get("id_map", {}))
                aggregated["failed_writes"].extend(result.get("failed_writes", []))
                aggregated["successful_writes"] += result.get("successful_writes", 0)
                failed_lines = result.get("failed_lines", [])
                if failed_lines:
                    aggregated["failed_lines"].extend(failed_lines)
                    if rpc_thread.writer and rpc_thread.fail_handle:
                        rpc_thread.writer.writerows(failed_lines)
                        rpc_thread.fail_handle.flush()  # Force write to disk

                error_summary = result.get("error_summary")
                if error_summary:
                    pretty_error = _format_odoo_error(error_summary)
                    rpc_thread.progress.console.print(
                        f"[bold red]Batch Error:[/bold red] {pretty_error}"
                    )

                rpc_thread.progress.update(rpc_thread.task_id, advance=1)

            except Exception as e:
                log.error(f"A worker thread failed unexpectedly: {e}", exc_info=True)
                rpc_thread.abort_flag = True
                rpc_thread.progress.console.print(
                    f"[bold red]Worker Failed: {e}[/bold red]"
                )
                rpc_thread.progress.update(
                    rpc_thread.task_id,
                    description="[bold red]FAIL:[/bold red] "
                    "Worker failed unexpectedly.",
                    refresh=True,
                )
                raise
            if rpc_thread.abort_flag:
                break
    except KeyboardInterrupt:
        log.warning("Ctrl+C detected! Aborting import gracefully...")
        rpc_thread.abort_flag = True
        rpc_thread.progress.console.print("[bold yellow]Aborted by user[/bold yellow]")
        rpc_thread.progress.update(
            rpc_thread.task_id,
            description="[bold yellow]Aborted by user[/bold yellow]",
            refresh=True,
        )
    finally:
        if futures and successful_batches == 0:
            log.error("Aborting import: All processed batches failed.")
            rpc_thread.abort_flag = True
        rpc_thread.executor.shutdown(wait=True, cancel_futures=True)
        rpc_thread.progress.update(
            rpc_thread.task_id,
            description=original_description,
            completed=rpc_thread.progress.tasks[rpc_thread.task_id].total,
        )

    return aggregated, rpc_thread.abort_flag


def _orchestrate_pass_1(
    progress: Progress,
    model_obj: Any,
    model_name: str,
    header: list[str],
    all_data: list[list[Any]],
    unique_id_field: str,
    deferred_fields: list[str],
    ignore: list[str],
    context: dict[str, Any],
    fail_writer: Optional[Any],
    fail_handle: Optional[TextIO],
    max_connection: int,
    batch_size: int,
    o2m: bool,
    split_by_cols: Optional[list[str]],
    force_create: bool = False,
) -> dict[str, Any]:
    """Orchestrates the multi-threaded Pass 1 (load/create).

    This function manages the first pass of the import process. It prepares
    the data by filtering out ignored and deferred fields, then executes the
    import in parallel using the `load` method with a `create` fallback.
    It is responsible for building the crucial ID map needed for Pass 2.

    Args:
        progress (Progress): The rich Progress instance for updating the UI.
        model_obj (Any): The connected Odoo model object used for RPC calls.
        model_name (str): The technical name of the target Odoo model.
        header (list[str]): The complete header from the source CSV file.
        all_data (list[list[Any]]): The complete data from the source CSV.
        unique_id_field (str): The name of the column containing the unique
            source ID for each record.
        deferred_fields (list[str]): A list of relational fields to ignore in
            this pass.
        ignore (list[str]): A list of additional fields to ignore, specified
            by the user.
        context (dict[str, Any]): The context dictionary for the Odoo RPC call.
        fail_writer (Optional[Any]): The CSV writer object for recording failures.
        fail_handle (Optional[TextIO]): The file handle for the fail file.
        max_connection (int): The number of parallel worker threads to use.
        batch_size (int): The number of records to process in each batch.
        o2m (bool): Enables one-to-many batching logic.
        force_create (bool): If True, bypasses the `load` method and uses
            the `create` method directly. Used for fail mode.
        split_by_cols: The column names to group records by to avoid concurrent updates.

    Returns:
        dict[str, Any]: A dictionary containing the results of the pass,
            including the `id_map` ({source_id: db_id}), a list of any
            `failed_lines`, and a `success` boolean flag.
    """
    rpc_pass_1 = RPCThreadImport(
        max_connection, progress, TaskID(0), fail_writer, fail_handle
    )
    pass_1_header, pass_1_data = header, all_data
    pass_1_ignore_list = deferred_fields + ignore

    try:
        pass_1_uid_index = pass_1_header.index(unique_id_field)
    except ValueError:
        log.error(
            f"Unique ID field '{unique_id_field}' was removed by the ignore list."
        )
        return {"success": False}

    pass_1_batches = list(
        _create_batches(pass_1_data, split_by_cols, pass_1_header, batch_size, o2m)
    )
    num_batches = len(pass_1_batches)
    pass_1_task = progress.add_task(
        f"Pass 1/2: Importing to [bold]{model_name}[/bold]",
        total=num_batches,
        last_error="",
    )
    rpc_pass_1.task_id = pass_1_task

    thread_state_1 = {
        "model": model_obj,
        "context": context,
        "unique_id_field_index": pass_1_uid_index,
        "batch_header": pass_1_header,
        "force_create": force_create,
        "progress": progress,
        "ignore_list": pass_1_ignore_list,
    }

    results, aborted = _run_threaded_pass(
        rpc_pass_1, _execute_load_batch, pass_1_batches, thread_state_1
    )
    results["success"] = not aborted
    return results


def _orchestrate_pass_2(
    progress: Progress,
    model_obj: Any,
    model_name: str,
    header: list[str],
    all_data: list[list[Any]],
    unique_id_field: str,
    id_map: dict[str, int],
    deferred_fields: list[str],
    context: dict[str, Any],
    fail_writer: Optional[Any],
    fail_handle: Optional[TextIO],
    max_connection: int,
    batch_size: int,
) -> tuple[bool, int]:
    """Orchestrates the multi-threaded Pass 2 (write).

    This function manages the second pass of a deferred import. It prepares
    the data for updating relational fields by using the ID map from Pass 1.
    It then groups records that have the exact same update payload and runs
    the `write` operations in parallel batches for maximum efficiency.

    Args:
        progress (Progress): The rich Progress instance for updating the UI.
        model_obj (Any): The connected Odoo model object.
        model_name (str): The technical name of the target Odoo model.
        header (list[str]): The header list from the original source file.
        all_data (list[list[Any]]): The full data from the original source file.
        unique_id_field (str): The name of the unique identifier column.
        id_map (dict[str, int]): The map of source IDs to database IDs from Pass 1.
        deferred_fields (list[str]): The list of fields to update in this pass.
        context (dict[str, Any]): The context dictionary for the Odoo RPC call.
        fail_writer (Optional[Any]): The CSV writer for the fail file.
        fail_handle (Optional[TextIO]): The file handle for the fail file.
        max_connection (int): The number of parallel worker threads to use.
        batch_size (int): The number of records per write batch.

    Returns:
        bool: True if the pass completed without any critical (abort-level)
        errors, False otherwise.
    """
    unique_id_field_index = header.index(unique_id_field)
    pass_2_data_to_write = _prepare_pass_2_data(
        all_data, header, unique_id_field_index, id_map, deferred_fields
    )

    if not pass_2_data_to_write:
        log.info("No valid relations found to update in Pass 2. Import complete.")
        return True, 0

    # --- Grouping Logic ---
    from collections import defaultdict

    grouped_writes = defaultdict(list)
    for db_id, vals in pass_2_data_to_write:
        # The key must be hashable, so we convert the dict to a frozenset of items.
        vals_key = frozenset(vals.items())
        grouped_writes[vals_key].append(db_id)

    # --- Batching Logic ---
    pass_2_batches = []
    for vals_key, ids in grouped_writes.items():
        vals = dict(vals_key)
        # Chunk the list of IDs into sub-batches of the desired size.
        for id_chunk in batch(ids, batch_size):
            pass_2_batches.append((list(id_chunk), vals))

    if not pass_2_batches:
        return True, 0

    num_batches = len(pass_2_batches)
    pass_2_task = progress.add_task(
        f"Pass 2/2: Updating [bold]{model_name}[/bold] relations",
        total=num_batches,
        last_error="",
    )
    rpc_pass_2 = RPCThreadImport(
        max_connection, progress, pass_2_task, fail_writer, fail_handle
    )
    thread_state_2 = {
        "model": model_obj,
        "progress": progress,
        "context": context,
    }
    pass_2_results, aborted = _run_threaded_pass(
        rpc_pass_2,
        _execute_write_batch,
        list(enumerate(pass_2_batches, 1)),
        thread_state_2,
    )

    failed_writes = pass_2_results.get("failed_writes", [])
    if fail_writer and failed_writes:
        log.warning("Writing failed Pass 2 records to fail file...")
        reverse_id_map = {v: k for k, v in id_map.items()}
        source_data_map = {row[unique_id_field_index]: row for row in all_data}
        failed_lines = []
        for db_id, _, error_message in failed_writes:
            source_id = reverse_id_map.get(db_id)
            if source_id and source_id in source_data_map:
                original_row = list(source_data_map[source_id])
                original_row.append(error_message)
                failed_lines.append(original_row)
        if failed_lines:
            fail_writer.writerows(failed_lines)

    # Pass 2 is successful ONLY if not aborted AND no writes failed.
    successful_writes = pass_2_results.get("successful_writes", 0)
    return not aborted and not failed_writes, successful_writes


def import_data(
    config: Union[str, dict[str, Any]],
    model: str,
    unique_id_field: str,
    file_csv: str,
    deferred_fields: Optional[list[str]] = None,
    context: Optional[dict[str, Any]] = None,
    fail_file: Optional[str] = None,
    encoding: str = "utf-8",
    separator: str = ";",
    ignore: Optional[list[str]] = None,
    max_connection: int = 1,
    batch_size: int = 10,
    skip: int = 0,
    force_create: bool = False,
    o2m: bool = False,
    split_by_cols: Optional[list[str]] = None,
) -> tuple[bool, dict[str, int]]:
    """Orchestrates a robust, multi-threaded, two-pass import process.

    This is the main entry point for the low-level import engine. It manages
    the entire workflow, including reading the source file, connecting to
    Odoo, and coordinating the import passes.

    The import is performed in one or two passes:
    - Pass 1: Creates base records using a multi-threaded `load` method with
      a `create` fallback for robustness. It builds a map of source IDs to
      new database IDs.
    - Pass 2: If `deferred_fields` are provided, it performs a second
      multi-threaded pass to `write` the relational data.

    Args:
        config (Union[str, dict]): Path to the Odoo connection file or a dict.
        model (str): The technical name of the target Odoo model.
        unique_id_field (str): The column name in the source file that
            uniquely identifies each record.
        file_csv (str): The full path to the source CSV data file.
        deferred_fields (Optional[list[str]]): A list of relational fields to
            process in a second pass. If None or empty, a single-pass
            import is performed.
        context (Optional[dict[str, Any]]): A context dictionary for Odoo
            RPC calls.
        fail_file (Optional[str]): Path to write any failed records to.
        encoding (str): The character encoding of the source file.
        separator (str): The delimiter character used in the source CSV.
        ignore (Optional[list[str]]): A list of columns to completely ignore
            from the source file.
        max_connection (int): The number of parallel threads to use.
        batch_size (int): The number of records to process in each batch.
        skip (int): The number of lines to skip at the top of the source file.
        force_create (bool): If True, bypasses the `load` method and uses
            the `create` method directly. Used for fail mode.
        o2m (bool): Enables special handling for one-to-many imports where
            child lines follow a parent record.
        split_by_cols: The column names to group records by to avoid concurrent updates.

    Returns:
        tuple[bool, int]: True if the entire import process completed without any
        critical, process-halting errors, False otherwise.
    """
    context, deferred, ignore = (
        context or {"tracking_disable": True},
        deferred_fields or [],
        ignore or [],
    )
    header, all_data = _read_data_file(file_csv, separator, encoding, skip)
    record_count = len(all_data)

    if not header:
        return False, {}

    try:
        if isinstance(config, dict):
            connection = conf_lib.get_connection_from_dict(config)
        else:
            connection = conf_lib.get_connection_from_config(config)
        model_obj = connection.get_model(model)
    except Exception as e:
        from .lib.internal.ui import _show_error_panel

        error_message = str(e)
        title = "Odoo Connection Error"
        friendly_message = (
            "Could not connect to Odoo. This usually means the connection "
            "details in your configuration file are incorrect.\n\n"
            "Please verify the following:\n"
            "  - [bold]hostname[/bold] is correct\n"
            "  - [bold]database[/bold] name is correct\n"
            "  - [bold]login[/bold] (username) is correct\n"
            "  - [bold]password[/bold] is correct\n\n"
            f"[bold]Original Error:[/bold] {error_message}"
        )
        _show_error_panel(title, friendly_message)
        return False, {}
    fail_writer, fail_handle = _setup_fail_file(fail_file, header, separator, encoding)
    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TextColumn("[green]{task.completed} of {task.total} batches"),
        "•",
        TimeElapsedColumn(),
        console=console,
        expand=True,
    )

    overall_success = False
    with progress:
        try:
            pass_1_results = _orchestrate_pass_1(
                progress,
                model_obj,
                model,
                header,
                all_data,
                unique_id_field,
                deferred,
                ignore,
                context,
                fail_writer,
                fail_handle,
                max_connection,
                batch_size,
                o2m,
                split_by_cols,
                force_create,
            )
            # A pass is only successful if it wasn't aborted.
            pass_1_successful = pass_1_results.get("success", False)
            if not pass_1_successful:
                return False, {}

            # If we get here, Pass 1 was not aborted. Now determine final status.
            id_map = pass_1_results.get("id_map", {})
            pass_2_successful = True  # Assume success if no Pass 2 is needed.
            updates_made = 0

            if deferred:
                pass_2_successful, updates_made = _orchestrate_pass_2(
                    progress,
                    model_obj,
                    model,
                    header,
                    all_data,
                    unique_id_field,
                    id_map,
                    deferred,
                    context,
                    fail_writer,
                    fail_handle,
                    max_connection,
                    batch_size,
                )

        finally:
            if fail_handle:
                fail_handle.close()

    overall_success = pass_1_successful and pass_2_successful
    stats = {
        "total_records": record_count,
        "created_records": len(id_map),
        "updated_relations": updates_made,
        "id_map": id_map,
    }
    return overall_success, stats
