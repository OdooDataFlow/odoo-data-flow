"""Export thread.

This module contains the low-level, multi-threaded logic for exporting
data from an Odoo instance.
"""

import concurrent.futures
import csv
import json
import shutil
import sys
from pathlib import Path
from time import time
from typing import Any, Optional, Union, cast

import httpx
import polars as pl
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .lib import cache, conf_lib
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
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """Splits the batch and recursively retries on network errors."""
        if len(ids_to_export) > 1:
            log.warning(
                f"Batch {num} failed with a network error ({e}). This is "
                "often a server timeout on large batches. Automatically "
                "splitting the batch and retrying."
            )
            mid_point = len(ids_to_export) // 2
            results_a, ids_a = self._execute_batch(
                ids_to_export[:mid_point], f"{num}-a"
            )
            results_b, ids_b = self._execute_batch(
                ids_to_export[mid_point:], f"{num}-b"
            )
            return results_a + results_b, ids_a + ids_b
        else:
            log.error(
                f"Export for record ID {ids_to_export[0]} in batch {num} "
                f"failed permanently after a network error: {e}"
            )
            self.has_failures = True
            return [], []

    def _execute_batch(
        self, ids_to_export: list[int], num: Union[int, str]
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """Executes the export for a single batch of IDs.

        This method attempts to fetch data for the given IDs. If it detects a
        network or memory error from the Odoo server, it splits the batch in
        half and calls itself recursively on the smaller sub-batches.

        Args:
            ids_to_export: A list of Odoo record IDs to export.
            num: The batch number, used for logging.

        Returns:
            A tuple containing:
            - A list of dictionaries representing the exported records.
            - A list of the database IDs that were successfully processed.
            Returns an empty list if the batch fails permanently.
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
                return [
                    dict(zip(self.header, row)) for row in exported_data
                ], ids_to_export

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
            # Ensure 'id' is always present for session tracking
            read_fields.add("id")

            # Fetch the raw data using the read method
            raw_data = cast(
                list[dict[str, Any]],
                self.model.read(ids_to_export, list(read_fields)),
            )
            if not raw_data:
                return [], []

            # Enrich with XML IDs if in hybrid mode
            if enrichment_tasks:
                self._enrich_with_xml_ids(raw_data, enrichment_tasks)

            processed_ids = [
                rec["id"] for rec in raw_data if isinstance(rec.get("id"), int)
            ]
            return self._format_batch_results(raw_data), processed_ids

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
                results_a, ids_a = self._execute_batch(
                    ids_to_export[:mid_point], f"{num}-a"
                )
                results_b, ids_b = self._execute_batch(
                    ids_to_export[mid_point:], f"{num}-b"
                )
                return results_a + results_b, ids_a + ids_b
            else:
                log.error(
                    f"Export for batch {num} failed permanently: {e}",
                    exc_info=True,
                )
                self.has_failures = True
                return [], []
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
    config: Union[str, dict[str, Any]],
    model_name: str,
    header: list[str],
    technical_names: bool,
) -> tuple[Optional[Any], Optional[Any], Optional[dict[str, dict[str, Any]]]]:
    """Connects to Odoo and fetches field metadata, including relations."""
    log.debug("Starting metadata initialization.")
    try:
        if isinstance(config, dict):
            connection = conf_lib.get_connection_from_dict(config)
        else:
            connection = conf_lib.get_connection_from_config(config)
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


def _enrich_main_df_with_xml_ids(
    df: pl.DataFrame, connection: Any, model_name: str
) -> pl.DataFrame:
    """Enriches a DataFrame with XML IDs for the main records.

    This function takes a DataFrame containing a '.id' column with numeric
    database IDs, fetches their corresponding external XML IDs from Odoo,
    and uses them to populate the 'id' column, preserving the '.id' column.

    Args:
        df: The Polars DataFrame to enrich. Must contain an '.id' column.
        connection: The active Odoo connection object.
        model_name: The name of the Odoo model being exported.

    Returns:
        The enriched DataFrame with the 'id' column populated with XML IDs
        and the '.id' column preserved.
    """
    if ".id" not in df.columns:
        log.warning("'.id' column not found, cannot perform main XML ID enrichment.")
        return df

    db_ids = df.get_column(".id").unique().drop_nulls().to_list()
    if not db_ids:
        log.debug("No database IDs found to enrich; ensuring 'id' is empty.")
        # Overwrite 'id' with nulls, keep '.id'
        return df.with_columns(pl.lit(None, dtype=pl.String).alias("id"))

    log.info(f"Fetching XML IDs for {len(db_ids)} main records...")
    ir_model_data = connection.get_model("ir.model.data")
    xml_id_data = ir_model_data.search_read(
        [("model", "=", model_name), ("res_id", "in", db_ids)],
        ["res_id", "module", "name"],
        context={"active_test": False},
    )

    if not xml_id_data:
        log.warning(f"No XML IDs found for the exported {model_name} records.")
        return df.with_columns(pl.lit(None, dtype=pl.String).alias("id"))

    df_xml_ids = (
        pl.from_dicts(xml_id_data)
        .with_columns(
            pl.format("{}.{}", pl.col("module"), pl.col("name")).alias("xml_id")
        )
        .select(pl.col("res_id").cast(pl.Int64), "xml_id")
        .unique(subset=["res_id"], keep="first")
    )

    # Join to get the xml_id, overwrite 'id', and drop temporary columns.
    df_enriched = df.join(df_xml_ids, left_on=".id", right_on="res_id", how="left")
    return df_enriched.with_columns(pl.col("xml_id").alias("id")).drop("xml_id")


def _process_export_batches(  # noqa: C901
    rpc_thread: "RPCThreadExport",
    total_ids: int,
    model_name: str,
    output: Optional[str],
    fields_info: dict[str, dict[str, Any]],
    separator: str,
    streaming: bool,
    session_dir: Optional[Path],
    is_resuming: bool,
    encoding: str,
    enrich_main_xml_id: bool = False,
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
                    batch_result, completed_ids = future.result()
                    if not batch_result:
                        continue

                    # --- Session State Update ---
                    if session_dir and completed_ids:
                        with (session_dir / "completed_ids.txt").open("a") as f:
                            for record_id in completed_ids:
                                f.write(f"{record_id}\n")
                    # --- End Session State Update ---

                    df = _clean_batch(batch_result)
                    if df.is_empty():
                        continue

                    final_batch_df = _clean_and_transform_batch(
                        df, field_types, polars_schema
                    )

                    if enrich_main_xml_id:
                        final_batch_df = _enrich_main_df_with_xml_ids(
                            final_batch_df, rpc_thread.connection, model_name
                        )

                    if output and streaming:
                        if not header_written:
                            if is_resuming:
                                with open(
                                    output, "a", newline="", encoding=encoding
                                ) as f:
                                    final_batch_df.write_csv(
                                        f,
                                        separator=separator,
                                        include_header=False,
                                    )
                            else:
                                final_batch_df.write_csv(
                                    output,
                                    separator=separator,
                                    include_header=True,
                                )
                            header_written = True
                        else:
                            with open(output, "a", newline="", encoding=encoding) as f:
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
        # Adjust schema for empty DataFrame if enrichment was active
        if enrich_main_xml_id:
            # The .id column is correctly typed as Int64. The id column, which
            # would also be Int64, needs its type changed to String for the header.
            polars_schema["id"] = pl.String()
        empty_df = pl.DataFrame(schema=polars_schema)
        if output:
            if is_resuming:
                with open(output, "a", newline="", encoding=encoding) as f:
                    empty_df.write_csv(f, separator=separator, include_header=False)
            else:
                empty_df.write_csv(output, separator=separator)
        return empty_df

    final_df = pl.concat(all_cleaned_dfs)
    if output:
        log.info(f"Writing {len(final_df)} records to {output}...")
        if is_resuming:
            with open(output, "a", newline="", encoding=encoding) as f:
                final_df.write_csv(f, separator=separator, include_header=False)
        else:
            final_df.write_csv(output, separator=separator)

        if not rpc_thread.has_failures:
            log.info("Export complete.")
    else:
        log.info("In-memory export complete.")
    return final_df


def _determine_export_strategy(
    config: Union[str, dict[str, Any]],
    model: str,
    header: list[str],
    technical_names: bool,
) -> tuple[
    Optional[Any],
    Optional[Any],
    Optional[dict[str, dict[str, Any]]],
    bool,
    bool,
    bool,
]:
    """Perform pre-flight checks and determine the best export strategy."""
    preliminary_read_mode = technical_names or any(
        f.endswith("/.id") or f == ".id" for f in header
    )
    connection, model_obj, fields_info = _initialize_export(
        config, model, header, preliminary_read_mode
    )

    if not model_obj or not fields_info:
        return None, None, None, False, False, False

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
        return None, None, None, False, False, False

    technical_types = {"selection", "binary"}
    has_technical_fields = any(
        info.get("type") in technical_types for info in fields_info.values()
    )
    is_hybrid = has_read_specifiers and has_xml_id_specifiers
    force_read_method = (
        technical_names or has_read_specifiers or is_hybrid or has_technical_fields
    )

    # --- New logic for main record XML ID enrichment ---
    enrich_main_xml_id = ".id" in header and "id" in header and force_read_method

    if enrich_main_xml_id:
        log.info(
            "Main record XML ID enrichment activated. "
            "'.id' will be used to fetch and populate 'id'."
        )
    elif is_hybrid:
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
            return None, None, None, False, False, False

    return (
        connection,
        model_obj,
        fields_info,
        force_read_method,
        is_hybrid,
        enrich_main_xml_id,
    )


def _resume_existing_session(
    session_dir: Path, session_id: str
) -> tuple[list[int], int]:
    """Resumes an existing export session by loading completed IDs."""
    log.info(f"Resuming export session: {session_id}")
    all_ids_file = session_dir / "all_ids.json"
    if not all_ids_file.exists():
        log.error(
            f"Session file 'all_ids.json' not found in {session_dir}. "
            "Cannot resume. Please start a new export."
        )
        return [], 0

    with all_ids_file.open("r") as f:
        all_ids = set(json.load(f))

    completed_ids_file = session_dir / "completed_ids.txt"
    completed_ids: set[int] = set()
    if completed_ids_file.exists():
        with completed_ids_file.open("r") as f:
            completed_ids = {int(line.strip()) for line in f if line.strip()}

    ids_to_export = list(all_ids - completed_ids)
    total_record_count = len(all_ids)

    log.info(
        f"{len(completed_ids)} of {total_record_count} records already "
        f"exported. Fetching remaining {len(ids_to_export)} records."
    )
    return ids_to_export, total_record_count


def _create_new_session(
    model_obj: Any,
    domain: list[Any],
    context: Optional[dict[str, Any]],
    session_id: str,
    session_dir: Path,
) -> tuple[list[int], int]:
    """Creates a new export session and fetches initial record IDs."""
    log.info(f"Starting new export session: {session_id}")
    log.info(f"Searching for records to export in model '{model_obj.model_name}'...")
    ids = model_obj.search(domain, context=context)
    total_record_count = len(ids)

    all_ids_file = session_dir / "all_ids.json"
    with all_ids_file.open("w") as f:
        json.dump(ids, f)
    (session_dir / "completed_ids.txt").touch()

    return ids, total_record_count


def export_data(
    config: Union[str, dict[str, Any]],
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
    resume_session: Optional[str] = None,
) -> tuple[bool, Optional[str], int, Optional[pl.DataFrame]]:
    """Exports data from an Odoo model, with support for resumable sessions."""
    session_id = resume_session or cache.generate_session_id(model, domain, header)
    session_dir = cache.get_session_dir(session_id)
    if not session_dir:
        return False, session_id, 0, None

    (
        connection,
        model_obj,
        fields_info,
        force_read_method,
        is_hybrid,
        enrich_main_xml_id,
    ) = _determine_export_strategy(config, model, header, technical_names)
    if not connection or not model_obj or not fields_info:
        return False, session_id, 0, None

    if streaming and not output:
        log.error("Streaming mode requires an output file path. Aborting.")
        return False, session_id, 0, None

    is_resuming = bool(resume_session)
    if is_resuming:
        ids_to_export, total_record_count = _resume_existing_session(
            session_dir, session_id
        )
    else:
        ids_to_export, total_record_count = _create_new_session(
            model_obj, domain, context, session_id, session_dir
        )

    if not ids_to_export:
        log.info("All records have already been exported. Nothing to do.")
        if output and not Path(output).exists():
            pl.DataFrame(schema=header).write_csv(output, separator=separator)
        if not is_resuming:
            shutil.rmtree(session_dir)
        return True, session_id, total_record_count, pl.DataFrame(schema=header)

    log.info(f"Processing {len(ids_to_export)} records in batches of {batch_size}.")
    id_batches = list(batch(ids_to_export, batch_size))

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

    final_df = _process_export_batches(
        rpc_thread,
        total_ids=total_record_count,
        model_name=model,
        output=output,
        fields_info=fields_info,
        separator=separator,
        streaming=streaming,
        session_dir=session_dir,
        is_resuming=is_resuming,
        encoding=encoding,
        enrich_main_xml_id=enrich_main_xml_id,
    )

    # --- Finalization and Cleanup ---
    success = not rpc_thread.has_failures
    if success:
        log.info("Export complete, cleaning up session directory.")
        shutil.rmtree(session_dir)
    else:
        log.error(f"Export failed. Session data retained in: {session_dir}")

    return success, session_id, total_record_count, final_df
