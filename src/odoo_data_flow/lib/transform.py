"""This module contains the core Processor class for transforming data."""

import inspect
import os
from collections import OrderedDict
from collections.abc import Mapping
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

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
        filename: Optional[str] = None,
        separator: str = ";",
        encoding: str = "utf-8",
        dataframe: Optional[pl.DataFrame] = None,
        preprocess: Callable[[pl.DataFrame], pl.DataFrame] = lambda df: df,
        **kwargs: Any,
    ) -> None:
        """Initializes the Processor."""
        self.file_to_write: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.dataframe: pl.DataFrame

        if filename:
            self.dataframe = self._read_file(filename, separator, encoding, **kwargs)
        elif dataframe is not None:
            self.dataframe = dataframe
        else:
            raise ValueError(
                "Processor must be initialized with either "
                "a 'filename' or a 'dataframe'."
            )

        self.dataframe = preprocess(self.dataframe)

    def _read_file(
        self, filename: str, separator: str, encoding: str, **kwargs: Any
    ) -> pl.DataFrame:
        """Reads a CSV or XML file and returns its content as a DataFrame."""
        _, file_extension = os.path.splitext(filename)
        xml_root_path = kwargs.get("xml_root_tag")

        if file_extension == ".csv":
            log.info(f"Reading CSV file: {filename}")
            try:
                return pl.read_csv(filename, separator=separator, encoding=encoding)
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
                    "An unexpected error occurred while reading XML file "
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
        grouped = self.dataframe.group_by(
            pl.struct(pl.all()).map_elements(
                lambda row: split_fun(row, 0), return_dtype=pl.Int64
            )
        )
        return {key: Processor(dataframe=group) for key, group in grouped}

    def get_o2o_mapping(self) -> dict[str, MapperRepr]:
        """Generates a direct 1-to-1 mapping dictionary."""
        return {
            str(column): MapperRepr(f"mapper.val('{column}')", mapper.val(column))
            for column in self.dataframe.columns
            if column
        }

    def process(
        self,
        mapping: Mapping[str, Union[Callable[..., Any], pl.Expr]],
        filename_out: str,
        params: Optional[dict[str, Any]] = None,
        t: str = "list",
        null_values: Optional[list[Any]] = None,
        m2m: bool = False,
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
            result_df = self._process_mapping_m2m(mapping, null_values=null_values)
        else:
            result_df = self._process_mapping(mapping, null_values=null_values)

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
                str_row = [str(item) for item in row]
                table.add_row(*str_row)

            console.print(table)
            log.info(f"Total rows that would be generated: {len(result_df)}")

            return result_df

        self._add_data(result_df, filename_out, params)
        return result_df

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
        encoding: str = "utf-8",
        dry_run: bool = False,
    ) -> None:
        """Joins data from a secondary file into the processor's main data.

        Args:
            filename: The path to the secondary file to join.
            master_key: The column name in the main data to join on.
            child_key: The column name in the secondary data to join on.
            header_prefix: A prefix to add to the headers from the child file.
            separator: The column separator for the child CSV file.
            encoding: The character encoding of the child file.
            dry_run: If True, prints a sample of the joined data to the
                console without modifying the processor's state.
        """
        child_df = self._read_file(filename, separator, encoding)
        child_df = child_df.rename(
            {col: f"{header_prefix}_{col}" for col in child_df.columns}
        )

        if dry_run:
            joined_df = self.dataframe.join(
                child_df, left_on=master_key, right_on=f"{header_prefix}_{child_key}"
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
                child_df, left_on=master_key, right_on=f"{header_prefix}_{child_key}"
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

    def _process_mapping(
        self,
        mapping: Mapping[str, Union[Callable[..., Any], pl.Expr]],
        null_values: list[Any],
    ) -> pl.DataFrame:
        """The core transformation loop."""
        import inspect

        state: dict[str, Any] = {}
        exprs = []

        def create_apply_func(
            func: Callable[..., Any],
            sig: "inspect.Signature",
            state: dict[str, Any],
        ) -> Callable[[dict[str, Any]], Any]:
            def apply_func(row: dict[str, Any]) -> Any:
                try:
                    if len(sig.parameters) == 1:
                        return func(row)
                    else:
                        return func(row, state)
                except SkippingError:
                    return ""

            return apply_func

        for key, func in mapping.items():
            if isinstance(func, pl.Expr):
                expr = func.alias(key)
            else:
                sig = inspect.signature(func)
                apply_func = create_apply_func(func, sig, state)
                expr = (
                    pl.struct(pl.all())
                    .map_elements(apply_func, return_dtype=pl.Object)
                    .alias(key)
                )
            exprs.append(expr)

        if not exprs:
            return pl.DataFrame()

        return self.dataframe.with_columns(exprs).drop_nulls()

    def _process_mapping_m2m(
        self,
        mapping: Mapping[str, Union[Callable[..., Any], pl.Expr]],
        null_values: list[Any],
    ) -> pl.DataFrame:
        """Handles special m2m mapping by expanding list values into unique rows."""
        result_df = self._process_mapping(mapping, null_values)

        list_cols = [
            col for col in result_df.columns if result_df[col].dtype == pl.List
        ]

        if not list_cols:
            return result_df

        return result_df.explode(list_cols)


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
            pl.DataFrame(attr_data, schema=attr_header), filename_out, import_args
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
        temp_processor = Processor(dataframe=unpivoted)
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
