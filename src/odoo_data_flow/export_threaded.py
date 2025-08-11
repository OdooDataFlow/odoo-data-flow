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
import httpx
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
while decrement:  # pragma: no cover
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
        connection: Any,
        model: Any,
        header: list[str],
        fields_info: dict[str, dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
        technical_names: bool = False,
        is_hybrid: bool = False,
    ) -> None:
        """Initializes the export thread handler.

        Args:
            max_connection: The maximum number of concurrent connections.
            connection: The odoolib connection object.
            model: The odoolib model object for making RPC calls.
            header: A list of field names to export.
            fields_info: A dictionary containing type and relation metadata.
            context: The Odoo context to use for the export.
            technical_names: If True, uses `model.read()` for raw database
                values.
            is_hybrid: If True, enables enrichment of `read` data with XML IDs.
        """
        super().__init__(max_connection)
        self.connection = connection
        self.model = model
        self.header = header
        self.fields_info = fields_info
        self.context = context or {}
        self.technical_names = technical_names
        self.is_hybrid = is_hybrid
        self.has_failures = False

    def _enrich_with_xml_ids(
        self,
        raw_data: list[dict[str, Any]],
        enrichment_tasks: list[dict[str, Any]],
    ) -> None:
        """Fetch XML IDs for related fields and enrich the raw_data in-place."""
        ir_model_data = self.connection.get_model("ir.model.data")
        for task in enrichment_tasks:
            relation_model = task["relation"]
            source_field = task["source_field"]
            if not relation_model or not isinstance(source_field, str):
                continue

            related_ids = list(
                {
                    rec[source_field][0]
                    for rec in raw_data
                    if isinstance(rec.get(source_field), (list, tuple))
                    and rec.get(source_field)
                }
            )
            if not related_ids:
                continue

            xml_id_data = ir_model_data.search_read(
                [("model", "=", relation_model), ("res_id", "in", related_ids)],
                ["res_id", "module", "name"],
            )
            db_id_to_xml_id = {
                item["res_id"]: f"{item['module']}.{item['name']}"
                for item in xml_id_data
            }

            for record in raw_data:
                related_val = record.get(source_field)
                xml_id = None
                if isinstance(related_val, (list, tuple)) and related_val:
                    xml_id = db_id_to_xml_id.get(related_val[0])
                record[task["target_field"]] = xml_id

    def _format_batch_results(
        self, raw_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Format the raw/enriched data to match the requested header."""
        processed_data = []
        for record in raw_data:
            new_record = {}
            for field in self.header:
                if field in record:
                    value = record[field]
                    if isinstance(value, (list, tuple)) and value:
                        new_record[field] = value[1]
                    else:
                        new_record[field] = value
                else:
                    base_field = field.split("/")[0].replace(".id", "id")
                    value = record.get(base_field)
                    if field == ".id":
                        new_record[".id"] = record.get("id")
                    elif field.endswith("/.id"):
                        new_record[field] = (
                            value[0]
                            if isinstance(value, (list, tuple)) and value
                            else None
                        )
                    else:
                        new_record[field] = None
            processed_data.append(new_record)
        return processed_data

    def _execute_batch_with_retry(
        self, ids_to_export: list[int], num: Union[int, str], e: Exception
    ) -> list[dict[str, Any]]:
        """Splits the batch and recursively retries on network errors."""
        if len(ids_to_export) > 1:
            log.warning(
                f"Batch {num} failed with a network error ({e}). This is "
                "often a server timeout on large batches. Automatically "
                "splitting the batch and retrying."
            )
            mid_point = len(ids_to_export) // 2
            results_a = self._execute_batch(ids_to_export[:mid_point], f"{num}-a")
            results_b = self._execute_batch(ids_to_export[mid_point:], f"{num}-b")
            return results_a + results_b
        else:
            log.error(
                f"Export for record ID {ids_to_export[0]} in batch {num} "
                f"failed permanently after a network error: {e}"
            )
            self.has_failures = True
            return []

    def _execute_batch(
        self, ids_to_export: list[int], num: Union[int, str]
    ) -> list[dict[str, Any]]:
        """Executes the export for a single batch of IDs.

        This method attempts to fetch data for the given IDs. If it detects a
        network or memory error from the Odoo server, it splits the batch in
        half and calls itself recursively on the smaller sub-batches.

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
            # Determine the fields to read and if enrichment is needed
            read_fields, enrichment_tasks = set(), []
            if not self.technical_names and not self.is_hybrid:
                # Use export_data for simple cases
                exported_data = self.model.export_data(
                    ids_to_export, self.header, context=self.context
                ).get("datas", [])
                return [dict(zip(self.header, row)) for row in exported_data]

            for field in self.header:
                base_field = field.split("/")[0].replace(".id", "id")
                read_fields.add(base_field)
                if self.is_hybrid and "/" in field and not field.endswith("/.id"):
                    enrichment_tasks.append(
                        {
                            "source_field": base_field,
                            "target_field": field,
                            "relation": self.fields_info[field].get("relation"),
                        }
                    )

            # Fetch the raw data using the read method
            raw_data = cast(
                list[dict[str, Any]],
                self.model.read(ids_to_export, list(read_fields)),
            )
            if not raw_data:
                return []

            # Enrich with XML IDs if in hybrid mode
            if enrichment_tasks:
                self._enrich_with_xml_ids(raw_data, enrichment_tasks)

            return self._format_batch_results(raw_data)

        except (
            httpx.ReadError,
            httpx.ReadTimeout,
        ) as e:
            # --- Resilient network error handling ---
            return self._execute_batch_with_retry(ids_to_export, num, e)

        except Exception as e:
            # --- MemoryError handling ---
            error_data = (
                e.args[0].get("data", {})
                if e.args and isinstance(e.args[0], dict)
                else {}
            )
            is_memory_error = error_data.get("name") == "builtins.MemoryError"
            if is_memory_error and len(ids_to_export) > 1:
                log.warning(
                    f"Batch {num} ({len(ids_to_export)} records) failed with "
                    f"MemoryError. Splitting and retrying..."
                )
                mid_point = len(ids_to_export) // 2
                results_a = self._execute_batch(ids_to_export[:mid_point], f"{num}-a")
                results_b = self._execute_batch(ids_to_export[mid_point:], f"{num}-b")
                return results_a + results_b
            else:
                log.error(
                    f"Export for batch {num} failed permanently: {e}",
                    exc_info=True,
                )
                self.has_failures = True
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


def _initialize_export(
    config_file: str, model_name: str, header: list[str], technical_names: bool
) -> tuple[Optional[Any], Optional[Any], Optional[dict[str, dict[str, Any]]]]:
    """Connects to Odoo and fetches field metadata, including relations."""
    log.debug("Starting metadata initialization.")
    try:
        connection = conf_lib.get_connection_from_config(config_file)
        model_obj = connection.get_model(model_name)
        fields_for_metadata = sorted(
            list(
                {f.split("/")[0].replace(".id", "id") for f in header if f != ".id"}
                | {"id"}
            )
        )
        field_metadata = model_obj.fields_get(fields_for_metadata)
        fields_info = {}
        for original_field in header:
            base_field = original_field.split("/")[0]
            meta = field_metadata.get(base_field)

            if not meta and original_field != ".id":
                log.warning(
                    f"Field '{original_field}' (base: '{base_field}') not found"
                    f" on model '{model_name}'. "
                    f"An empty column will be created."
                )

            field_type = "char"
            if meta:
                field_type = meta["type"]
            if original_field == ".id" or original_field.endswith("/.id"):
                field_type = "integer"
            elif original_field == "id":
                field_type = "integer" if technical_names else "char"
            fields_info[original_field] = {"type": field_type}
            if meta and meta.get("relation"):
                fields_info[original_field]["relation"] = meta["relation"]
        log.debug(f"Successfully initialized metadata. Fields info: {fields_info}")
        return connection, model_obj, fields_info
    except Exception as e:
        log.error(f"Failed during metadata initialization. Error: {e}", exc_info=True)
        return None, None, None


def _clean_batch(batch_data: list[dict[str, Any]]) -> pl.DataFrame:
    """Converts a batch of data to a DataFrame without complex cleaning."""
    if not batch_data:
        return pl.DataFrame()
    return pl.DataFrame(batch_data, infer_schema_length=None)


def _clean_and_transform_batch(
    df: pl.DataFrame,
    field_types: dict[str, str],
    polars_schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    """Runs a multi-stage cleaning and transformation pipeline on a DataFrame."""
    # Step 1: Convert any list-type or object-type columns to strings FIRST.
    transform_exprs = []
    for col_name in df.columns:
        if df[col_name].dtype in (pl.List, pl.Object):
            transform_exprs.append(pl.col(col_name).cast(pl.String))
    if transform_exprs:
        df = df.with_columns(transform_exprs)

    # Step 2: Now that lists are gone, it's safe to clean up 'False' values.
    false_cleaning_exprs = []
    for field_name, field_type in field_types.items():
        if field_name in df.columns and field_type != "boolean":
            false_cleaning_exprs.append(
                pl.when(pl.col(field_name) == False)  # noqa: E712
                .then(None)
                .otherwise(pl.col(field_name))
                .alias(field_name)
            )
    if false_cleaning_exprs:
        df = df.with_columns(false_cleaning_exprs)

    # Step 3: Handle boolean string conversions.
    bool_cols_to_convert = [
        k
        for k, v in polars_schema.items()
        if v.base_type() == pl.Boolean and k in df.columns and df[k].dtype != pl.Boolean
    ]
    if bool_cols_to_convert:
        conversion_exprs = [
            pl.when(
                pl.col(c)
                .cast(pl.String, strict=False)
                .str.to_lowercase()
                .is_in(["true", "1", "t", "yes"])
            )
            .then(True)
            .otherwise(False)
            .alias(c)
            for c in bool_cols_to_convert
        ]
        df = df.with_columns(conversion_exprs)

    # Step 4: Ensure all schema columns exist before the final cast.
    for col_name in polars_schema:
        if col_name not in df.columns:
            df = df.with_columns(
                pl.lit(None, dtype=polars_schema[col_name]).alias(col_name)
            )

    # Step 5: Final cast to the target schema.
    casted_df = df.cast(polars_schema, strict=False)  # type: ignore[arg-type]
    return casted_df.select(list(polars_schema.keys()))


def _process_export_batches(  # noqa: C901
    rpc_thread: "RPCThreadExport",
    total_ids: int,
    model_name: str,
    output: Optional[str],
    fields_info: dict[str, dict[str, Any]],
    separator: str,
    streaming: bool,
) -> Optional[pl.DataFrame]:
    """Processes exported batches.

    Uses streaming for large files if requested,
    otherwise concatenates in memory for best performance.
    """
    field_types = {k: v.get("type", "char") for k, v in fields_info.items()}
    polars_schema: dict[str, pl.DataType] = {
        field: ODOO_TO_POLARS_MAP.get(odoo_type, pl.String)()
        for field, odoo_type in field_types.items()
    }
    if polars_schema:
        polars_schema = {
            k: v() if isinstance(v, type) and issubclass(v, pl.DataType) else v
            for k, v in polars_schema.items()
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
    try:
        with progress:
            task = progress.add_task(
                f"[cyan]Exporting {model_name}...", total=total_ids
            )
            for future in concurrent.futures.as_completed(rpc_thread.futures):
                try:
                    batch_result = future.result()
                    if not batch_result:
                        continue

                    df = _clean_batch(batch_result)
                    if df.is_empty():
                        continue

                    final_batch_df = _clean_and_transform_batch(
                        df, field_types, polars_schema
                    )

                    if output and streaming:
                        if not header_written:
                            final_batch_df.write_csv(
                                output, separator=separator, include_header=True
                            )
                            header_written = True
                        else:
                            with open(output, "ab") as f:
                                final_batch_df.write_csv(
                                    f, separator=separator, include_header=False
                                )
                    else:
                        all_cleaned_dfs.append(final_batch_df)
                    progress.update(task, advance=len(batch_result))
                except Exception as e:
                    log.error(f"A task in a worker thread failed: {e}", exc_info=True)
                    rpc_thread.has_failures = True
    except KeyboardInterrupt:  # pragma: no cover
        log.warning("\nExport process interrupted by user. Shutting down workers...")
        rpc_thread.executor.shutdown(wait=True, cancel_futures=True)
        log.error("Export aborted.")
        return None

    rpc_thread.executor.shutdown(wait=True)

    if rpc_thread.has_failures:
        log.error(
            "Export finished with errors. Some records could not be exported. "
            "Please check the logs above for details on failed records."
        )
    if output and streaming:
        log.info(f"Streaming export complete. Data written to {output}")
        return None
    if not all_cleaned_dfs:
        log.warning("No data was returned from the export.")
        empty_df = pl.DataFrame(schema=polars_schema)
        if output:
            empty_df.write_csv(output, separator=separator)
        return empty_df

    final_df = pl.concat(all_cleaned_dfs)
    if output:
        log.info(f"Writing {len(final_df)} records to {output}...")
        final_df.write_csv(output, separator=separator)
        if not rpc_thread.has_failures:
            log.info("Export complete.")
    else:
        log.info("In-memory export complete.")
    return final_df


def _determine_export_strategy(
    config_file: str, model: str, header: list[str], technical_names: bool
) -> tuple[
    Optional[Any],
    Optional[Any],
    Optional[dict[str, dict[str, Any]]],
    bool,
    bool,
]:
    """Perform pre-flight checks and determine the best export strategy."""
    preliminary_read_mode = technical_names or any(
        f.endswith("/.id") or f == ".id" for f in header
    )
    connection, model_obj, fields_info = _initialize_export(
        config_file, model, header, preliminary_read_mode
    )

    if not model_obj or not fields_info:
        return None, None, None, False, False

    has_read_specifiers = any(f.endswith("/.id") or f == ".id" for f in header)
    has_xml_id_specifiers = any(f.endswith("/id") for f in header)
    has_other_subfield_specifiers = any(
        "/" in f and not f.endswith("/id") and not f.endswith("/.id") for f in header
    )

    if has_read_specifiers and has_other_subfield_specifiers:
        invalid_fields = [
            f
            for f in header
            if "/" in f and not f.endswith("/id") and not f.endswith("/.id")
        ]
        log.error(
            "Mixing raw ID specifiers (e.g., '.id') with relational sub-fields "
            f"(e.g., {invalid_fields}) is not supported in hybrid mode. "
            "Only 'field/id' is allowed for enrichment."
        )
        return None, None, None, False, False

    technical_types = {"selection", "binary"}
    has_technical_fields = any(
        info.get("type") in technical_types for info in fields_info.values()
    )
    is_hybrid = has_read_specifiers and has_xml_id_specifiers
    force_read_method = (
        technical_names or has_read_specifiers or is_hybrid or has_technical_fields
    )

    if is_hybrid:
        log.info("Hybrid export mode activated. Using 'read' with XML ID enrichment.")
    elif has_technical_fields:
        log.info("Read method auto-enabled for 'selection' or 'binary' fields.")
    elif force_read_method:
        log.info("Exporting using 'read' method for raw database values.")
    else:
        log.info("Exporting using 'export_data' method for human-readable values.")

    if force_read_method and not is_hybrid:
        invalid_fields = [f for f in header if "/" in f and not f.endswith("/.id")]
        if invalid_fields:
            log.error(
                f"Mixing export-style specifiers {invalid_fields} "
                f"is not supported in pure 'read' mode."
            )
            return None, None, None, False, False

    return connection, model_obj, fields_info, force_read_method, is_hybrid


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
    connection, model_obj, fields_info, force_read_method, is_hybrid = (
        _determine_export_strategy(config_file, model, header, technical_names)
    )

    if not connection or not model_obj or not fields_info:
        return None

    if streaming and not output:
        log.error("Streaming mode requires an output file path. Aborting.")
        return None

    log.info(
        f"Searching for records to export in model '{model}' with domain: \n{domain}"
    )
    ids = model_obj.search(domain, context=context)
    if not ids:
        log.warning("No records found for the given domain.")
        if output:
            pl.DataFrame(schema=header).write_csv(output, separator=separator)
        return None

    log.info(
        f"Found {len(ids)} records to export. Splitting into batches of {batch_size}."
    )
    id_batches = list(batch(ids, batch_size))

    rpc_thread = RPCThreadExport(
        max_connection=max_connection,
        connection=connection,
        model=model_obj,
        header=header,
        fields_info=fields_info,
        context=context,
        technical_names=force_read_method,
        is_hybrid=is_hybrid,
    )
    for i, id_batch in enumerate(id_batches):
        rpc_thread.launch_batch(list(id_batch), i)

    return _process_export_batches(
        rpc_thread, len(ids), model, output, fields_info, separator, streaming
    )
