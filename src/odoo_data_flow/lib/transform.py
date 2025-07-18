"""This module contains the core Processor class for transforming data."""

import inspect
import os
import sys
from collections import OrderedDict
from collections.abc import Mapping
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

from .odoo_lib import build_polars_schema

if sys.version_info < (3, 10):
    from typing import Union as TypeUnion
else:
    TypeUnion = Union

if sys.version_info < (3, 10):
    from typing import Union as TypeUnion
else:
    TypeUnion = Union

import polars as pl
from lxml import etree  # type: ignore[import-untyped]
from rich.console import Console
from rich.table import Table

from ..logging_config import log
from . import mapper
from .internal.exceptions import SkippingError
from .internal.io import write_file


class MapperRepr:
    """A wrapper to provide a useful string representation for mapper functions."""

    def __init__(self, repr_string: str, func: Callable[..., Any]) -> None:
        """Initializes the MapperRepr.

        Args:
            repr_string: The string representation to use for the mapper.
            func: The actual callable mapper function.
        """
        self._repr_string = repr_string
        self.func = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calls the wrapped mapper function."""
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        """Returns the custom string representation."""
        return self._repr_string


class Processor:
    """Core class for reading, transforming, and preparing data for Odoo."""

    def __init__(
        self,
        mapping: Mapping[str, Any],
        source_filename: Optional[str] = None,
        dataframe: Optional[pl.DataFrame] = None,
        connection: Optional[Any] = None,
        model: Optional[str] = None,
        config_file: Optional[str] = None,
        separator: str = ";",
        preprocess: Callable[[pl.DataFrame], pl.DataFrame] = lambda df: df,
        schema_overrides: Optional[dict[str, pl.DataType]] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the Processor.

        The Processor can be initialized either by providing a `source_filename`
        to read from disk, or by providing a `dataframe` to work with in-memory
        data.

        Args:
            source_filename: The path to the source CSV or XML file.
            config_file: Path to the Odoo connection configuration file. Used as
                        a fallback for operations that require a DB
                        connection if no specific config is provided later.
            model: The Odoo model name (e.g., 'product.template').
            dataframe: A Polars DataFrame to initialize the Processor with.
            connection: An optional Odoo connection object for schema fetching.
            model: The Odoo model name (e.g., 'product.template'). If provided
                with a connection, a base schema will be pre-fetched.
            config_file: Path to the Odoo connection configuration file.
            separator: The column delimiter for CSV files.
            preprocess: A function to modify the raw data (Polars DataFrame)
                before mapping begins.
            connection: An optional Odoo connection object.
            mapping: A dictionary defining the transformation rules. Can also,
                    include Polars data types as tuples for manual schema
                    overrides e.g. `'field': (pl.Utf8, mapper_function)`.
            schema_overrides: A dictionary mapping column names to Polars data
                            types to optimize CSV reading performance. This is
                            the recommended way to provide a schema for
                            offline processing.
            **kwargs: Catches other arguments, primarily for XML processing.
        """
        self.file_to_write: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.dataframe: pl.DataFrame
        self.config_file = config_file

        manual_overrides, self.logic_mapping = self._parse_mapping(mapping)

        final_schema: dict[str, pl.DataType] = {}
        if connection and model:
            schema_from_odoo = build_polars_schema(connection, model)
            final_schema.update({k: v() for k, v in schema_from_odoo.items()})
        if schema_overrides:
            final_schema.update(schema_overrides)
        if manual_overrides:
            final_schema.update(manual_overrides)

        # --- FINAL NORMALIZATION STEP ---
        # Ensure all values in the final schema are instances, not classes.
        # This resolves all mypy/typeguard errors downstream.
        if final_schema:
            final_schema = {
                k: v()
                if inspect.isclass(v) and issubclass(v, pl.DataType)  # type: ignore[unreachable]
                else v
                for k, v in final_schema.items()
            }
        # --- END FINAL NORMALIZATION STEP ---

        self.schema_overrides = final_schema if final_schema else None

        if source_filename:
            self.dataframe = self._read_file(
                source_filename, separator, self.schema_overrides, **kwargs
            )
        elif dataframe is not None:
            self.dataframe = dataframe
        else:
            raise ValueError(
                "Processor must be initialized with either "
                "a 'source_filename' or a 'dataframe'."
            )

        self.dataframe = preprocess(self.dataframe)

    def _parse_mapping(
        self, mapping: Optional[Mapping[str, Any]]
    ) -> tuple[dict[str, pl.DataType], dict[str, Any]]:
        """Parses a mapping dict to separate Polars types from mapper functions."""
        schema_overrides: dict[str, pl.DataType] = {}
        logic_mapping: dict[str, Any] = {}
        if not mapping:
            return schema_overrides, logic_mapping

        for key, value in mapping.items():
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], type)
                and issubclass(value[0], pl.DataType)
            ):
                schema_overrides[key] = value[0]()  # Instantiate the type
                logic_mapping[key] = value[1]
            else:
                # It's a standard mapping: function
                logic_mapping[key] = value
        return schema_overrides, logic_mapping

    def _read_file(
        self,
        filename: str,
        separator: str,
        schema_overrides: Optional[dict[str, pl.DataType]] = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Reads a CSV or XML file and returns its content as a DataFrame."""
        _, file_extension = os.path.splitext(filename)
        xml_root_path = kwargs.get("xml_root_tag")

        if file_extension == ".csv":
            log.info(f"Reading CSV file: {filename}")
            try:
                return pl.read_csv(
                    filename,
                    separator=separator,
                    encoding="utf-8",
                    schema_overrides=schema_overrides,
                    try_parse_dates=True,
                )
            except Exception as e:
                log.error(f"Failed to read CSV file {filename}: {e}")
                return pl.DataFrame()
        elif xml_root_path:
            log.info(f"Reading XML file: {filename}")
            try:
                parser = etree.XMLParser(
                    resolve_entities=False,
                    no_network=True,
                    dtd_validation=False,
                    load_dtd=False,
                )
                tree = etree.parse(filename, parser=parser)
                if kwargs.get("xml_record_tag"):
                    nodes = tree.xpath(f"//{kwargs.get('xml_record_tag')}")
                else:
                    nodes = tree.xpath(xml_root_path)

                if not nodes:
                    log.warning(f"No nodes found for root path '{xml_root_path}'")
                    return pl.DataFrame()

                data = [{elem.tag: elem.text for elem in node} for node in nodes]
                return pl.DataFrame(data)
            except etree.XMLSyntaxError as e:
                log.error(f"Failed to parse XML file {filename}: {e}")
                return pl.DataFrame()
            except Exception as e:
                log.error(
                    f"An unexpected error occurred while reading XML file "
                    f"{filename}: {e}"
                )
                return pl.DataFrame()
        return pl.DataFrame()

    def check(
        self, check_fun: Callable[..., bool], message: Optional[str] = None
    ) -> bool:
        """Runs a data quality check function against the loaded data.

        Args:
            check_fun: The checker function to execute.
            message: An optional custom error message to display on failure.

        Returns:
            True if the check passes, False otherwise.
        """
        res = check_fun(self.dataframe)
        if not res:
            error_message = (
                message or f"Data quality check '{check_fun.__name__}' failed."
            )
            log.warning(error_message)
        return res

    def split(self, split_fun: Callable[..., Any]) -> dict[Any, "Processor"]:
        """Splits the processor's data into multiple new Processor objects.

        Args:
            split_fun: A function that takes a row dictionary and index, and
                returns a key to group the row by.

        Returns:
            A dictionary where keys are the grouping keys and values are new
            Processor instances containing the grouped data.
        """
        # Group by the key and create new processors
        df_with_index = self.dataframe.with_row_index()

        grouped = df_with_index.group_by(
            pl.struct(pl.all()).map_elements(
                lambda row: str(split_fun(row, row.get("index"))),
                return_dtype=pl.String,
            )
        )
        schema_items = (self.schema_overrides or {}).items()
        new_mapping = {
            **self.logic_mapping,
            **{
                k: (v, self.logic_mapping.get(k))
                for k, v in schema_items
                if self.logic_mapping.get(k)
            },
        }

        return {
            str(key[0]): Processor(
                mapping=new_mapping,
                # 3. Drop the temporary index column from the final grouped DataFrames.
                dataframe=group.drop("index"),
            )
            for key, group in grouped
        }

    def get_o2o_mapping(self) -> dict[str, MapperRepr]:
        """Generates a direct 1-to-1 mapping dictionary."""
        return {
            str(column): MapperRepr(f"mapper.val('{column}')", mapper.val(column))
            for column in self.dataframe.columns
            if column
        }

    def process(
        self,
        filename_out: str,
        params: Optional[dict[str, Any]] = None,
        t: str = "list",
        null_values: Optional[list[Any]] = None,
        m2m: bool = False,
        m2m_columns: Optional[list[str]] = None,
        dry_run: bool = False,
    ) -> pl.DataFrame:
        """Processes the data using a mapping and prepares it for writing.

        Args:
            mapping: The mapping dictionary defining the transformation rules.
            filename_out: The path where the output CSV file will be saved.
            params: A dictionary of parameters for the `odoo-data-flow import`
                command, used when generating the load script.
            t: The type of collection to return data in ('list' or 'set').
            null_values: A list of values to be treated as empty.
            m2m: If True, activates special processing for many-to-many data.
            m2m_columns: A list of column names to unpivot when `m2m=True`.
            dry_run: If True, prints a sample of the output to the console
                instead of writing files.


        Returns:
            A Dataframe containing the header list and the transformed data.
        """
        if null_values is None:
            null_values = ["NULL", False]
        if params is None:
            params = {}

        result_df: pl.DataFrame
        if m2m:
            result_df = self._process_mapping_m2m(
                self.logic_mapping,
                null_values=null_values,
                m2m_columns=m2m_columns,
            )
        else:
            result_df = self._process_mapping(
                self.logic_mapping, null_values=null_values
            )

        if t == "set":
            result_df = result_df.unique()

        if dry_run:
            console = Console()
            log.info("--- DRY RUN MODE (Outputting sample of first 10 rows) ---")
            log.info("No files will be written.")

            table = Table(title="Dry Run Output Sample")
            for column_header in result_df.columns:
                table.add_column(column_header, style="cyan")

            for row in result_df.head(10).iter_rows():
                table.add_row(*(str(item) for item in row))
            console.print(table)
            log.info(f"Total rows that would be generated: {len(result_df)}")

            return result_df

        self._add_data(result_df, filename_out, params)
        return result_df

    def process_m2m(
        self,
        id_column: str,
        m2m_columns: list[str],
        filename_out: str,
        params: Optional[dict[str, Any]] = None,
        separator: str = ",",
    ) -> None:
        """Processes many-to-many data by first unpivoting the source data.

        This is a robust alternative to using the m2m=True flag. It unnests
        comma-separated values from the 'm2m_columns' into individual rows
        before processing.

        Args:
            id_column: The column to use as the stable ID (e.g., 'id' or 'ref').
            m2m_columns: A list of columns that contain the m2m values.
            filename_out: The path where the output CSV file will be saved.
            params: A dictionary of parameters for the `odoo-data-flow import` command.
            separator: The separator for the values within the m2m columns.
        """
        log.info(f"Processing m2m data for columns: {m2m_columns}")

        # 1. Explode the comma-separated strings into lists of strings
        df_with_lists = self.dataframe.with_columns(
            [pl.col(c).str.split(separator) for c in m2m_columns]
        )

        # 2. Explode the DataFrame on the list columns to create new rows
        exploded_df = df_with_lists.explode(m2m_columns)

        # 3. Create a new processor with this "tidy" data and the stored mapping
        m2m_processor = Processor(mapping=self.logic_mapping, dataframe=exploded_df)

        # 4. Process the data (no m2m=True flag needed) and add to write queue
        result_df = m2m_processor.process(
            filename_out=filename_out, params=params, t="set"
        )
        self._add_data(result_df, filename_out, params or {})

    def write_to_file(
        self,
        script_filename: str,
        fail: bool = True,
        append: bool = False,
        python_exe: str = "python",
        path: str = "",
    ) -> None:
        """Generates the .sh script for the import.

        Args:
            script_filename: The path where the shell script will be saved.
            fail: If True, includes a second command with the --fail flag.
            append: If True, appends to the script file instead of overwriting.
            python_exe: The python executable to use in the script.
            path: The path to prepend to the odoo-data-flow command.
        """
        init = not append
        for _, info in self.file_to_write.items():
            info_copy = info.copy()
            # NEW: Use the config from params if available,
            #  otherwise use the processor's default
            info_copy["conf_file"] = str(info.get("config")) or self.config_file
            info_copy.update(
                {
                    "model": info.get("model", "auto"),
                    "init": init,
                    "launchfile": script_filename,
                    "fail": fail,
                    "python_exe": python_exe,
                    "path": path,
                }
            )
            write_file(**info_copy)
            init = False

    def join_file(
        self,
        filename: str,
        master_key: str,
        child_key: str,
        header_prefix: str = "child",
        separator: str = ";",
        schema_overrides: Optional[dict[str, pl.DataType]] = None,
        dry_run: bool = False,
    ) -> None:
        """Joins data from a secondary file into the processor's main data.

        Args:
            filename: The path to the secondary file to join.
            master_key: The column name in the main data to join on.
            child_key: The column name in the secondary data to join on.
            header_prefix: A prefix to add to the headers from the child file.
            separator: The column separator for the child CSV file.
            schema_overrides: A dictionary to override Polars' inferred data
                types for the joined file.
            dry_run: If True, prints a sample of the joined data to the
                console without modifying the processor's state.
        """
        log.info(f"Joining with secondary file: {filename}")
        child_df = self._read_file(filename, separator, schema_overrides)

        # This part correctly renames all columns EXCEPT the join key
        child_df = child_df.rename(
            {
                col: f"{header_prefix}_{col}"
                for col in child_df.columns
                if col != child_key
            }
        )

        if dry_run:
            joined_df = self.dataframe.join(
                child_df, left_on=master_key, right_on=child_key
            )
            log.info("--- DRY RUN MODE (Outputting sample of joined data) ---")
            console = Console()
            table = Table(title="Joined Data Sample")

            for column_header in joined_df.columns:
                table.add_column(column_header, style="cyan")

            for row in joined_df.head(10).iter_rows():
                str_row = [str(item) for item in row]
                table.add_row(*str_row)

            console.print(table)
            log.info(f"Total rows that would be generated: {len(joined_df)}")
        else:
            self.dataframe = self.dataframe.join(
                child_df, left_on=master_key, right_on=child_key
            )

    def _add_data(
        self,
        dataframe: pl.DataFrame,
        filename_out: str,
        params: dict[str, Any],
    ) -> None:
        """Adds data to the internal write queue."""
        params_copy = params.copy()
        params_copy["filename"] = (
            os.path.abspath(filename_out) if filename_out else False
        )
        params_copy["dataframe"] = dataframe
        self.file_to_write[filename_out] = params_copy

    def _cast_result_for_polars(self, result: Any, dtype: pl.DataType) -> Any:
        """Casts a Python object to a type suitable for a Polars column.

        This helper handles robust type conversion for data returned from
        mapper functions before it's passed back into a Polars Series.

        Args:
            result: The value returned from a mapper function.
            dtype: The target Polars DataType for the column.

        Returns:
            The casted value, or None if casting fails.
        """
        if dtype.is_integer():
            try:
                return int(result)
            except (ValueError, TypeError):
                return None
        if dtype.is_float():
            try:
                return float(result)
            except (ValueError, TypeError):
                return None
        if isinstance(dtype, pl.Boolean):
            if isinstance(result, str):
                return result.lower() in ("true", "1", "t", "yes")
            return bool(result)
        if isinstance(dtype, pl.String):
            return str(result)
        return result

    def _process_mapping(
        self,
        mapping: Mapping[str, Union[Callable[..., Any], pl.Expr]],
        null_values: list[Any],
        list_return_dtype: Optional[pl.DataType] = None,
    ) -> pl.DataFrame:
        """The core transformation loop."""
        if not mapping:
            return self.dataframe.clone()

        exprs = []
        state: dict[str, Any] = {}

        for key, func in mapping.items():
            if isinstance(func, pl.Expr):
                # This branch handles Polars expressions. It's now self-contained.
                expr = func.alias(key)
                exprs.append(expr)
            else:
                # This branch handles callable functions.
                # All its logic is now inside the 'else'.
                sig = inspect.signature(func)
                target_dtype = (
                    self.schema_overrides.get(key, pl.String())
                    if self.schema_overrides
                    else pl.String()
                )

                def _create_wrapper(
                    f: Callable[..., Any],
                    signature: inspect.Signature,
                    dtype: pl.DataType,
                ) -> Callable[[dict[str, Any]], Any]:
                    def wrapper(row_struct: dict[str, Any]) -> Any:
                        try:
                            result = (
                                f(row_struct, state)
                                if len(signature.parameters) > 1
                                else f(row_struct)
                            )
                            if result is None:
                                return None
                            return self._cast_result_for_polars(result, dtype)
                        except SkippingError:
                            return None

                    return wrapper

                if isinstance(target_dtype, type) and issubclass(
                    target_dtype, pl.DataType
                ):
                    resolved_target_dtype = target_dtype()
                else:
                    resolved_target_dtype = target_dtype

                wrapper_func = _create_wrapper(func, sig, resolved_target_dtype)
                unique_cols = list(dict.fromkeys(self.dataframe.columns))

                expr = (
                    pl.struct(unique_cols)
                    .map_elements(
                        wrapper_func,
                        return_dtype=resolved_target_dtype,
                    )
                    .alias(key)
                )
                exprs.append(expr)

        if not exprs:
            return pl.DataFrame()

        return self.dataframe.select(exprs)

    def _process_mapping_m2m(
        self,
        mapping: Mapping[str, Union[Callable[..., Any], pl.Expr]],
        null_values: list[Any],
        m2m_columns: Optional[list[str]],
    ) -> pl.DataFrame:
        """Specific m2m processor.

        Handles m2m mapping by unpivoting data to a long format internally
        before applying a simple mapping.
        """
        if not m2m_columns:
            raise ValueError(
                "The 'm2m_columns' argument must be provided when m2m=True."
            )

        # 1. Unpivot the specified columns to create a robust long-format DataFrame
        id_vars = [col for col in self.dataframe.columns if col not in m2m_columns]
        unpivoted_df = self.dataframe.unpivot(
            index=id_vars,
            on=m2m_columns,
            variable_name="m2m_source_column",  # e.g., 'Color', 'Size_H'
            value_name="m2m_source_value",  # e.g., 'Blue', 'L'
        ).filter(pl.col("m2m_source_value").is_not_null())

        # 2. Create a temporary processor with this simple, unpivoted data
        temp_processor = Processor(mapping=mapping, dataframe=unpivoted_df)

        # 3. Apply the original mapping logic, which now works on simple rows
        return temp_processor._process_mapping(mapping, null_values=null_values)


class ProductProcessorV10(Processor):
    """Processor for the modern (Odoo v13+) product attribute model."""

    def process_attribute_data(
        self,
        attributes_list: list[str],
        attribute_prefix: str,
        filename_out: str,
        import_args: dict[str, Any],
    ) -> None:
        """Creates and registers the `product.attribute.csv` file.

        Args:
            attributes_list: A list of attribute names (e.g., ['Color', 'Size']).
            attribute_prefix: The prefix for generating external IDs.
            filename_out: The output path for the CSV file.
            import_args: A dictionary of parameters for the import script.
        """
        attr_header = ["id", "name", "create_variant"]
        attr_data = [
            {
                "id": mapper.to_m2o(attribute_prefix, att),
                "name": att,
                "create_variant": "Dynamically",
            }
            for att in attributes_list
        ]
        # Corrected: Use the 'schema' argument to enforce column order.
        self._add_data(
            pl.DataFrame(attr_data, schema=attr_header),
            filename_out,
            import_args,
        )

    # NEW METHOD for product.attribute.value.csv
    def process_attribute_value_data(
        self,
        attribute_list: list[str],
        attribute_value_prefix: str,
        attribute_prefix: str,
        filename_out: str,
        import_args: dict[str, Any],
    ) -> None:
        """Collects unique attribute values.

        Collects unique attribute values from all product lines and prepares them
        for product.attribute.value.csv.

        Args:
            attribute_list: List of attribute column names (e.g., ['Color', 'Size_H']).
            attribute_value_prefix: The XML ID prefix for attribute values.
            attribute_prefix: The XML ID prefix for attributes themselves.
            filename_out: The output path for product.attribute.value.csv.
            import_args: Import parameters for the script.
        """
        melted_df = self.dataframe.unpivot(
            index=[col for col in self.dataframe.columns if col not in attribute_list],
            on=attribute_list,
            variable_name="attribute_name",
            value_name="value",
        )
        unique_values = (
            melted_df.filter(pl.col("value").is_not_null())
            .unique(subset=["attribute_name", "value"])
            .select(
                pl.struct(["attribute_name", "value"])
                .map_elements(
                    lambda row: mapper.to_m2o(
                        attribute_value_prefix,
                        f"{row['attribute_name']}_{row['value']}",
                    ),
                    return_dtype=pl.String,
                )
                .alias("id"),
                pl.col("value").alias("name"),
                pl.col("attribute_name")
                .map_elements(
                    lambda name: mapper.to_m2o(attribute_prefix, name),
                    return_dtype=pl.String,
                )
                .alias("attribute_id/id"),
            )
        )
        self._add_data(unique_values, filename_out, import_args)


class ProductProcessorV9(Processor):
    """Processor for the legacy (Odoo v9-v12) product attribute model."""

    def _generate_attribute_file_data(
        self, attributes_list: list[str], prefix: str
    ) -> pl.DataFrame:
        """Generates a DataFrame for 'product.attribute.csv'."""
        attr_data = [
            {"id": mapper.to_m2o(prefix, attr), "name": attr}
            for attr in attributes_list
        ]
        return pl.DataFrame(attr_data)

    def _extract_attribute_value_data(
        self,
        mapping: Mapping[str, Union[pl.Expr, Callable[..., Any]]],
        attributes_list: list[str],
    ) -> pl.DataFrame:
        """Extracts and transforms data for 'product.attribute.value.csv'."""
        id_cols = [col for col in self.dataframe.columns if col not in attributes_list]
        unpivoted = self.dataframe.unpivot(
            index=id_cols,
            on=attributes_list,
            variable_name="attribute_name",
            value_name="attribute_value_name",
        ).filter(pl.col("attribute_value_name").is_not_null())

        # Create a temporary processor to reuse the robust mapping logic
        temp_processor = Processor(dataframe=unpivoted, mapping=mapping)
        result_df = temp_processor._process_mapping(mapping, null_values=[])
        return result_df.unique()

    def process_attribute_mapping(
        self,
        mapping: Mapping[str, Union[pl.Expr, Callable[..., Any]]],
        line_mapping: Mapping[str, Union[pl.Expr, Callable[..., Any]]],
        attributes_list: list[str],
        attribute_prefix: str,
        path: str,
        import_args: dict[str, Any],
    ) -> None:
        """Orchestrates the processing of legacy product attributes."""
        # 1. Generate product.attribute.csv
        attr_df = self._generate_attribute_file_data(attributes_list, attribute_prefix)
        self._add_data(attr_df, path + "product.attribute.csv", import_args)

        # 2. Generate product.attribute.value.csv
        values_df = self._extract_attribute_value_data(mapping, attributes_list)
        self._add_data(values_df, path + "product.attribute.value.csv", import_args)

        # 3. Generate product.attribute.line.csv
        line_df = self._process_mapping(line_mapping, null_values=[])
        line_import_args = dict(import_args, groupby="product_tmpl_id/id")
        self._add_data(line_df, path + "product.attribute.line.csv", line_import_args)
