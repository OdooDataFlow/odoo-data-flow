"""Export thread.

This module contains the low-level, multi-threaded logic for exporting
data from an Odoo instance.
"""

import concurrent.futures
import csv
import sys
from time import time
from typing import Any, Optional, Union, cast

import polars as pl
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
from .lib.odoo_lib import ODOO_TO_POLARS_MAP
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
    """Export Thread handler with automatic batch resizing on MemoryError.

    This class manages worker threads for exporting data from Odoo. It includes
    a fallback mechanism that automatically splits and retries batches if the
    Odoo server runs out of memory processing a large request.
    """

    def __init__(
        self,
        max_connection: int,
        model: Any,
        header: list[str],
        context: Optional[dict[str, Any]] = None,
        technical_names: bool = False,
    ) -> None:
        """Initializes the export thread handler.

        Args:
            max_connection: The maximum number of concurrent connections.
            model: The odoolib model object for making RPC calls.
            header: A list of field names to export.
            context: The Odoo context to use for the export.
            technical_names: If True, uses `model.read()` for technical field
                names. Otherwise, uses `model.export_data()`.
        """
        super().__init__(max_connection)
        self.model = model
        self.header = header
        self.context = context or {}
        self.technical_names = technical_names

    def _execute_batch(
        self, ids_to_export: list[int], num: Union[int, str]
    ) -> list[dict[str, Any]]:
        """Executes the export for a single batch of IDs.

        This method attempts to fetch data for the given IDs. If it detects a
        MemoryError from the Odoo server, it splits the batch in half and
        calls itself recursively on the smaller sub-batches.

        Args:
            ids_to_export: A list of Odoo record IDs to export.
            num: The batch number, used for logging.

        Returns:
            A list of dictionaries representing the exported records. Returns an
            empty list if the batch fails permanently.
        """
        start_time = time()
        log.debug(f"Exporting batch {num} with {len(ids_to_export)} records...")
        try:
            if self.technical_names:
                return cast(
                    list[dict[str, Any]],
                    self.model.read(ids_to_export, self.header),
                )
            else:
                exported_data = self.model.export_data(
                    ids_to_export, self.header, context=self.context
                ).get("datas", [])
                return [dict(zip(self.header, row)) for row in exported_data]

        except Exception as e:
            error_data = e.args[0].get("data", {}) if e.args else {}
            is_memory_error = error_data.get("name") == "builtins.MemoryError"

            if is_memory_error and len(ids_to_export) > 1:
                log.warning(
                    f"Batch {num} ({len(ids_to_export)} records) "
                    "failed with MemoryError. "
                    "Splitting batch and retrying..."
                )
                mid_point = len(ids_to_export) // 2
                batch_a = ids_to_export[:mid_point]
                batch_b = ids_to_export[mid_point:]

                results_a = self._execute_batch(batch_a, f"{num}-a")
                results_b = self._execute_batch(batch_b, f"{num}-b")

                return results_a + results_b
            else:
                log.error(
                    f"Export for batch {num} failed permanently: {e}",
                    exc_info=True,
                )
                return []
        finally:
            log.debug(f"Batch {num} finished in {time() - start_time:.2f}s.")

    def launch_batch(self, data_ids: list[int], batch_number: int) -> None:
        """Submits a batch of IDs to be exported by a worker thread.

        Args:
            data_ids: The list of record IDs to process in this batch.
            batch_number: The sequential number of this batch.
        """
        self.spawn_thread(self._execute_batch, [data_ids, batch_number])


def _clean_batch(
    batch_data: list[dict[str, Any]], field_types: dict[str, str]
) -> pl.DataFrame:
    """Converts a batch of data to a DataFrame and cleans it."""
    if not batch_data:
        return pl.DataFrame()

    df = pl.DataFrame(batch_data, infer_schema_length=None)
    cleaning_exprs = []
    for field_name, field_type in field_types.items():
        if field_name in df.columns and field_type != "boolean":
            cleaning_exprs.append(
                pl.when(pl.col(field_name) == False)  # noqa: E712
                .then(None)
                .otherwise(pl.col(field_name))
                .alias(field_name)
            )
    if cleaning_exprs:
        df = df.with_columns(cleaning_exprs)
    return df


def _initialize_export(
    config_file: str, model_name: str, header: list[str]
) -> tuple[Optional[Any], Optional[dict[str, str]]]:
    """Connects to Odoo and fetches field metadata."""
    try:
        connection = conf_lib.get_connection_from_config(config_file)
        model_obj = connection.get_model(model_name)
        field_metadata = model_obj.fields_get(header)
        field_types = {
            field: details["type"] for field, details in field_metadata.items()
        }
        return model_obj, field_types
    except Exception as e:
        log.error(
            f"Failed to connect to Odoo or get model '{model_name}'. "
            f"Please check your configuration. Error: {e}"
        )
        return None, None


def _handle_completed_batch(
    future: concurrent.futures.Future[list[dict[str, Any]]],
    field_types: dict[str, str],
    polars_schema: dict[str, pl.DataType],
    output: Optional[str],
    streaming: bool,
    header_written: bool,
    separator: str,
) -> tuple[Optional[pl.DataFrame], bool]:
    """Processes a single completed export batch from a worker thread.

    Args:
        future: The Future object whose result needs to be processed.
        field_types: A dictionary mapping Odoo field names to their types.
        polars_schema: The target Polars schema for casting.
        output: The output file path, if any.
        streaming: A boolean indicating if streaming mode is active.
        header_written: A boolean indicating if the CSV header has been written.
        separator: The CSV separator character.

    Returns:
        A tuple containing the processed DataFrame (or None if streamed) and
        the updated header_written flag.
    """
    try:
        batch_result = future.result()
        if not batch_result:
            return None, header_written

        cleaned_df = _clean_batch(batch_result, field_types)
        if cleaned_df.is_empty():
            return None, header_written

        # Add a type ignore comment to handle the complex type hint mismatch
        casted_df = cleaned_df.cast(polars_schema, strict=False)  # type: ignore[arg-type]

        if output and streaming:
            if not header_written:
                casted_df.write_csv(
                    output, separator=separator, include_header=True
                )
                header_written = True
            else:
                with open(output, "ab") as f:
                    casted_df.write_csv(
                        f, separator=separator, include_header=False
                    )
            return None, header_written
        else:
            return casted_df, header_written
    except Exception as e:
        log.error(f"A task in a worker thread failed: {e}", exc_info=True)
        return None, header_written


def _finalize_export(
    all_dfs: list[pl.DataFrame],
    field_types: dict[str, str],
    output: Optional[str],
    separator: str,
) -> Optional[pl.DataFrame]:
    """Finalizes the export after all batches are processed.

    This function concatenates DataFrames for in-memory mode, writes the
    final result to a file if needed, and handles the case of no data.

    Args:
        all_dfs: A list of all processed batch DataFrames.
        field_types: A dictionary mapping Odoo field names to their types.
        output: The output file path, if any.
        separator: The CSV separator character.

    Returns:
        The final, complete DataFrame, or None if in streaming mode.
    """
    if not all_dfs:
        log.warning("No data was returned from the export.")
        empty_df = pl.DataFrame(schema=list(field_types.keys()))
        if output:
            empty_df.write_csv(output, separator=separator)
        return empty_df

    final_df = pl.concat(all_dfs)
    if output:
        log.info(f"Writing {len(final_df)} records to {output}...")
        final_df.write_csv(output, separator=separator)
        log.info("Export complete.")
    else:
        log.info("In-memory export complete.")

    return final_df


def _process_export_batches(
    rpc_thread: RPCThreadExport,
    total_ids: int,
    model_name: str,
    output: Optional[str],
    field_types: dict[str, str],
    separator: str,
    streaming: bool,
) -> Optional[pl.DataFrame]:
    """Orchestrates the processing of exported batches.

    This function initializes schemas and progress bars, then iterates through
    completed worker threads, delegating processing to helper functions.

    Args:
        rpc_thread: The RPCThreadExport instance managing worker threads.
        total_ids: The total number of records to be exported.
        model_name: The name of the Odoo model being exported.
        output: The path to the output file, if specified.
        field_types: A dictionary mapping field names to their Odoo types.
        separator: The character to use as a separator in the CSV file.
        streaming: If True, enables streaming mode to save memory.

    Returns:
        A Polars DataFrame containing all exported data, or None if in
        streaming mode with a file output.
    """
    polars_schema: dict[str, pl.DataType] = {
        field: ODOO_TO_POLARS_MAP.get(odoo_type, pl.String)()
        for field, odoo_type in field_types.items()
    }
    all_cleaned_dfs: list[pl.DataFrame] = []
    header_written = False

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
    with progress:
        task = progress.add_task(
            f"[cyan]Exporting {model_name}...", total=total_ids
        )
        for future in concurrent.futures.as_completed(rpc_thread.futures):
            processed_df, header_written = _handle_completed_batch(
                future,
                field_types,
                polars_schema,
                output,
                streaming,
                header_written,
                separator,
            )
            if processed_df is not None:
                all_cleaned_dfs.append(processed_df)

            # Heuristic to advance progress even if batch fails internally
            # This part is tricky as future.result() is inside the helper
            # For simplicity, we assume success or failure of the whole batch
            # A more advanced solution would require more state passing
            progress.update(
                task, advance=rpc_thread.executor._work_queue.qsize()
            )

    rpc_thread.executor.shutdown(wait=True)

    if output and streaming:
        log.info(f"Streaming export complete. Data written to {output}")
        return None

    return _finalize_export(all_cleaned_dfs, field_types, output, separator)


def export_data(
    config_file: str,
    model: str,
    domain: list[Any],
    header: list[str],
    output: Optional[str],
    context: Optional[dict[str, Any]] = None,
    max_connection: int = 1,
    batch_size: int = 1000,
    separator: str = ";",
    encoding: str = "utf-8",
    technical_names: bool = False,
    streaming: bool = False,
) -> Optional[pl.DataFrame]:
    """Exports data from an Odoo model."""
    model_obj, field_types = _initialize_export(config_file, model, header)
    if not model_obj or field_types is None:
        return None

    if streaming and not output:
        log.error("Streaming mode requires an output file path. Aborting.")
        return None

    log.info(f"Searching for records in model '{model}' to export...")
    ids = model_obj.search(domain, context=context)
    if not ids:
        log.warning("No records found for the given domain.")
        if output:
            pl.DataFrame(schema=header).write_csv(output, separator=separator)
        return pl.DataFrame(schema=header)

    log.info(
        f"Found {len(ids)} records to export. Splitting into batches of {batch_size}."
    )
    id_batches = list(batch(ids, batch_size))

    rpc_thread = RPCThreadExport(
        max_connection, model_obj, header, context, technical_names
    )
    for i, id_batch in enumerate(id_batches):
        rpc_thread.launch_batch(list(id_batch), i)

    return _process_export_batches(
        rpc_thread, len(ids), model, output, field_types, separator, streaming
    )
