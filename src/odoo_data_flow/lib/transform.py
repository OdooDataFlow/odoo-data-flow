"""This module contains the core Processor class for transforming data."""

import csv
import os
from collections import OrderedDict
from typing import Any, Callable, Optional, Union

from lxml import etree  # type: ignore[import-untyped]

from ..logging_config import log
from . import mapper
from .internal.exceptions import SkippingError
from .internal.io import write_file
from .internal.tools import AttributeLineDict


class MapperRepr:
    """Mapper representation.

    A wrapper to provide a useful string representation for mapper functions.
    """

    def __init__(self, repr_string: str, func: Callable[..., Any]) -> None:
        """Initializes the MapperRepr."""
        self._repr_string = repr_string
        self.func = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function."""
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        """Return the string representation."""
        return self._repr_string


class Processor:
    """Core class for reading, transforming, and preparing data for Odoo."""

    def __init__(
        self,
        filename: Optional[str] = None,
        separator: str = ";",
        encoding: str = "utf-8",
        header: Optional[list[str]] = None,
        data: Optional[list[list[Any]]] = None,
        preprocess: Callable[
            [list[str], list[list[Any]]], tuple[list[str], list[list[Any]]]
        ] = lambda h, d: (h, d),
        **kwargs: Any,
    ) -> None:
        """Initializes the Processor."""
        self.file_to_write: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.header: list[str]
        self.data: list[list[Any]]

        # Determine if initializing from a file or in-memory data
        if filename:
            self.header, self.data = self._read_file(
                filename, separator, encoding, **kwargs
            )
        elif header is not None and data is not None:
            self.header = header
            self.data = data
        else:
            raise ValueError(
                "Processor must be initialized with either a 'filename' or both"
                " 'header' and 'data'."
            )

        # Apply any pre-processing hooks
        self.header, self.data = preprocess(self.header, self.data)

    def _read_file(
        self, filename: str, separator: str, encoding: str, **kwargs: Any
    ) -> tuple[list[str], list[list[Any]]]:
        """Reads a CSV or XML file and returns its header and data."""
        xml_root_path = kwargs.get("xml_root_tag")

        if xml_root_path:
            log.info(f"Reading XML file: {filename}")
            try:
                parser = etree.XMLParser(
                    resolve_entities=False,
                    no_network=True,
                    dtd_validation=False,
                    load_dtd=False,
                )
                tree = etree.parse(filename, parser=parser)  # noqa: S320
                nodes = tree.xpath(xml_root_path)

                if not nodes:
                    log.warning(f"No nodes found for root path '{xml_root_path}'")
                    return [], []

                header = [elem.tag for elem in nodes[0]]
                data = []
                for node in nodes:
                    row = [
                        (child.text if child is not None else "")
                        for child in (node.find(col) for col in header)
                    ]
                    data.append(row)
                return header, data

            except etree.XMLSyntaxError as e:
                log.error(f"Failed to parse XML file {filename}: {e}")
            except Exception as e:
                log.error(
                    "An unexpected error occurred while reading XML file "
                    f"{filename}: {e}"
                )
            return [], []
        else:
            log.info(f"Reading CSV file: {filename}")
            try:
                with open(filename, encoding=encoding, newline="") as f:
                    reader = csv.reader(f, delimiter=separator)
                    header = next(reader)
                    data = [row for row in reader]
                    return header, data
            except FileNotFoundError:
                log.error(f"Source file not found at: {filename}")
            except Exception as e:
                log.error(f"Failed to read file {filename}: {e}")
        return [], []

    def check(
        self, check_fun: Callable[..., bool], message: Optional[str] = None
    ) -> bool:
        """Runs a data quality check function against the loaded data."""
        res = check_fun(self.header, self.data)
        if not res:
            error_message = (
                message or f"Data quality check '{check_fun.__name__}' failed."
            )
            log.warning(error_message)
        return res

    def split(self, split_fun: Callable[..., Any]) -> dict[Any, "Processor"]:
        """Splits the processor's data into multiple new Processor objects."""
        grouped_data: OrderedDict[Any, list[list[Any]]] = OrderedDict()
        for i, row in enumerate(self.data):
            row_dict = dict(zip(self.header, row))
            key = split_fun(row_dict, i)
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(row)

        return {
            key: Processor(header=list(self.header), data=data)
            for key, data in grouped_data.items()
        }

    def get_o2o_mapping(self) -> dict[str, MapperRepr]:
        """Generates a direct 1-to-1 mapping dictionary."""
        return {
            str(column): MapperRepr(f"mapper.val('{column}')", mapper.val(column))
            for column in self.header
            if column
        }

    def process(
        self,
        mapping: dict[str, Callable[..., Any]],
        filename_out: str,
        params: Optional[dict[str, Any]] = None,
        t: str = "list",
        null_values: Optional[list[Any]] = None,
        m2m: bool = False,
    ) -> tuple[list[str], Union[list[Any], set[tuple[Any, ...]]]]:
        """Main processor.

        Processes the data using a mapping and prepares it for writing.
        """
        if null_values is None:
            null_values = ["NULL", False]
        if params is None:
            params = {}

        head: list[str]
        data: Union[list[Any], set[tuple[Any, ...]]]
        if m2m:
            head, data = self._process_mapping_m2m(mapping, null_values=null_values)
        else:
            head, data = self._process_mapping(mapping, t=t, null_values=null_values)

        self._add_data(head, data, filename_out, params)
        return head, data

    def write_to_file(
        self,
        script_filename: str,
        fail: bool = True,
        append: bool = False,
        python_exe: str = "python",
        path: str = "",
    ) -> None:
        """Write bash script.

        Generates the .sh script for the import.
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
    ) -> None:
        """File joiner.

        Joins data from a secondary file into the processor's main data.
        """
        child_header, child_data = self._read_file(filename, separator, encoding)

        try:
            child_key_pos = child_header.index(child_key)
            master_key_pos = self.header.index(master_key)
        except ValueError as e:
            log.error(
                f"Join key error: {e}. Check if '{master_key}' and "
                f"'{child_key}' exist in their respective files."
            )
            return

        child_data_map = {row[child_key_pos]: row for row in child_data}

        empty_child_row = [""] * len(child_header)
        for master_row in self.data:
            key_value = master_row[master_key_pos]
            row_to_join = child_data_map.get(key_value, empty_child_row)
            master_row.extend(row_to_join)

        self.header.extend([f"{header_prefix}_{h}" for h in child_header])

    def _add_data(
        self,
        head: list[str],
        data: Union[list[Any], set[tuple[Any, ...]]],
        filename_out: str,
        params: dict[str, Any],
    ) -> None:
        """Adds data to the write queue."""
        params_copy = params.copy()
        params_copy["filename"] = (
            os.path.abspath(filename_out) if filename_out else False
        )
        params_copy["header"] = head
        params_copy["data"] = data
        self.file_to_write[filename_out] = params_copy

    def _process_mapping(
        self,
        mapping: dict[str, Callable[..., Any]],
        t: str,
        null_values: list[Any],
    ) -> tuple[list[str], Union[list[Any], set[tuple[Any, ...]]]]:
        """The core transformation loop."""
        lines_out: Union[list[Any], set[tuple[Any, ...]]] = [] if t == "list" else set()
        state: dict[str, Any] = {}

        for i, line in enumerate(self.data):
            cleaned_line = [
                s.strip() if s and s.strip() not in null_values else "" for s in line
            ]
            line_dict = dict(zip(self.header, cleaned_line))

            try:
                line_out = [mapping[k](line_dict, state) for k in mapping.keys()]
            except SkippingError as e:
                log.debug(f"Skipping line {i}: {e.message}")
                continue
            except TypeError:
                line_out = [mapping[k](line_dict) for k in mapping.keys()]

            if isinstance(lines_out, list):
                lines_out.append(line_out)
            else:
                lines_out.add(tuple(line_out))
        return list(mapping.keys()), lines_out

    def _process_mapping_m2m(
        self,
        mapping: dict[str, Callable[..., Any]],
        null_values: list[Any],
    ) -> tuple[list[str], list[Any]]:
        """m2m process mapping.

        Handles special m2m mapping by expanding list values into unique rows.
        """
        head, data_unioned = self._process_mapping(mapping, "list", null_values)
        data = list(data_unioned)  # Ensure data is a list
        lines_out: list[Any] = []

        for line_out in data:
            index_list, zip_list = [], []
            for index, value in enumerate(line_out):
                if isinstance(value, list):
                    index_list.append(index)
                    zip_list.append(value)

            if not zip_list:
                if line_out not in lines_out:
                    lines_out.append(line_out)
                continue

            values_list = zip(*zip_list)
            for values in values_list:
                new_line = list(line_out)
                for i, val in enumerate(values):
                    new_line[index_list[i]] = val
                if new_line not in lines_out:
                    lines_out.append(new_line)

        return head, lines_out


class ProductProcessorV10(Processor):
    """Processor for generating a 'product.attribute' file."""

    def process_attribute_data(
        self,
        attributes_list: list[str],
        attribute_prefix: str,
        filename_out: str,
        import_args: dict[str, Any],
    ) -> None:
        """Creates and registers the 'product.attribute.csv' file."""
        attr_header = ["id", "name", "create_variant"]
        attr_data = [
            [mapper.to_m2o(attribute_prefix, att), att, "Dynamically"]
            for att in attributes_list
        ]
        self._add_data(attr_header, attr_data, filename_out, import_args)


class ProductProcessorV9(Processor):
    """Processor to generate variant data from a flat file.

    Creates three CSV files:
    1. product.attribute.csv: The attributes themselves.
    2. product.attribute.value.csv: The specific values for each attribute.
    3. product.attribute.line.csv: Links attributes to product templates.
    """

    def _generate_attribute_file_data(
        self, attributes_list: list[str], prefix: str
    ) -> tuple[list[str], list[list[str]]]:
        """Generates header and data for 'product.attribute.csv'."""
        header = ["id", "name"]
        data = [[mapper.to_m2o(prefix, attr), attr] for attr in attributes_list]
        return header, data

    def _extract_attribute_value_data(
        self,
        mapping: dict[str, Callable[..., Any]],
        attributes_list: list[str],
        processed_rows: list[dict[str, Any]],
    ) -> set[tuple[Any, ...]]:
        """Extracts and transforms data for 'product.attribute.value.csv'."""
        attribute_values: set[tuple[Any, ...]] = set()
        name_key = "name"

        for row_dict in processed_rows:
            try:
                line_out_results = [mapping[k](row_dict) for k in mapping.keys()]
            except TypeError:
                line_out_results = [mapping[k](row_dict, {}) for k in mapping.keys()]

            name_mapping_index = list(mapping.keys()).index(name_key)
            values_dict = line_out_results[name_mapping_index]

            if not isinstance(values_dict, dict):
                continue

            for attr_name in attributes_list:
                if values_dict.get(attr_name):
                    value_line = tuple(
                        res[attr_name] if isinstance(res, dict) else res
                        for res in line_out_results
                    )
                    attribute_values.add(value_line)

        return attribute_values

    def process_attribute_mapping(
        self,
        mapping: dict[str, Callable[..., Any]],
        line_mapping: dict[str, Callable[..., Any]],
        attributes_list: list[str],
        attribute_prefix: str,
        path: str,
        import_args: dict[str, Any],
        id_gen_fun: Optional[Callable[..., str]] = None,
        null_values: Optional[list[str]] = None,
    ) -> None:
        """Orchestrates the processing of product attributes and variants."""
        _null_values = null_values if null_values is not None else ["NULL"]
        attr_header, attr_data = self._generate_attribute_file_data(
            attributes_list, attribute_prefix
        )

        processed_rows: list[dict[str, Any]] = []
        for line in self.data:
            cleaned_line = [
                s.strip() if s and s.strip() not in _null_values else "" for s in line
            ]
            processed_rows.append(dict(zip(self.header, cleaned_line)))

        values_header = list(mapping.keys())
        values_data = self._extract_attribute_value_data(
            mapping, attributes_list, processed_rows
        )

        _id_gen_fun = id_gen_fun or (
            lambda tmpl_id, vals: mapper.to_m2o(
                tmpl_id.split(".")[0] + "_LINE", tmpl_id
            )
        )
        line_aggregator = AttributeLineDict(attr_data, _id_gen_fun)
        for row_dict in processed_rows:
            try:
                values_lines = [line_mapping[k](row_dict) for k in line_mapping.keys()]
            except TypeError:
                values_lines = [
                    line_mapping[k](row_dict, {}) for k in line_mapping.keys()
                ]
            line_aggregator.add_line(values_lines, list(line_mapping.keys()))
        line_header, line_data = line_aggregator.generate_line()

        context = import_args.setdefault("context", {})
        context["create_product_variant"] = True

        self._add_data(
            attr_header, attr_data, path + "product.attribute.csv", import_args
        )
        self._add_data(
            values_header,
            values_data,
            path + "product.attribute.value.csv",
            import_args,
        )

        line_import_args = dict(import_args, groupby="product_tmpl_id/id")
        self._add_data(
            line_header,
            line_data,
            path + "product.attribute.line.csv",
            line_import_args,
        )
