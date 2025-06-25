"""This module contains the core Processor class for transforming data."""

import csv
import os
from collections import OrderedDict
from typing import Callable, Optional

from lxml import etree

from ..logging_config import log
from . import mapper
from .internal.exceptions import SkippingError
from .internal.io import write_file
from .internal.tools import AttributeLineDict


class MapperRepr:
    """Mapper representation.

    A wrapper to provide a useful string representation for mapper functions.
    """

    def __init__(self, repr_string, func):
        self._repr_string = repr_string
        self.func = func

    def __call__(self, *args, **kwargs):
        """Call the wrapped function."""
        return self.func(*args, **kwargs)

    def __repr__(self):
        """Return the string representation."""
        return self._repr_string


class Processor:
    """Core class for reading, transforming, and preparing data for Odoo."""

    def __init__(
        self,
        filename=None,
        separator=";",
        encoding="utf-8",
        header=None,
        data=None,
        preprocess=lambda h, d: (h, d),
        **kwargs,
    ):
        self.file_to_write = OrderedDict()

        # Determine if initializing from a file or in-memory data
        if filename:
            # The 'xml_...' kwargs are passed to the file reader
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

    def _read_file(self, filename, separator, encoding, **kwargs):
        """Reads a CSV or XML file and returns its header and data."""
        xml_root_path = kwargs.get("xml_root_tag")

        if xml_root_path:
            log.info(f"Reading XML file: {filename}")
            try:
                # Use a secure parser to prevent XXE and other vulnerabilities
                parser = etree.XMLParser(
                    resolve_entities=False,
                    no_network=True,
                    dtd_validation=False,
                    load_dtd=False,
                )
                tree = etree.parse(filename, parser=parser)
                nodes = tree.xpath(xml_root_path)

                if not nodes:
                    log.warning(
                        f"No nodes found for root path '{xml_root_path}'"
                    )
                    return [], []

                # Infer header from the tags of the first node's children
                header = [elem.tag for elem in nodes[0]]
                data = []
                for node in nodes:
                    row = []
                    for col in header:
                        # Find the child element and get its text content
                        child = node.find(col)
                        row.append(child.text if child is not None else "")
                    data.append(row)
                return header, data

            except etree.XMLSyntaxError as e:
                log.error(f"Failed to parse XML file {filename}: {e}")
                return [], []
            except Exception as e:
                log.error(
                    f"An unexpected error occurred while reading XML file {filename}: {e}"
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
                return [], []
            except Exception as e:
                log.error(f"Failed to read file {filename}: {e}")
                return [], []

    def check(self, check_fun, message=None):
        """Runs a data quality check function against the loaded data."""
        res = check_fun(self.header, self.data)
        if not res:
            error_message = (
                message or f"Data quality check '{check_fun.__name__}' failed."
            )
            log.warning(error_message)
        return res

    def split(self, split_fun):
        """Splits the processor's data into multiple new Processor objects."""
        grouped_data = OrderedDict()
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

    def get_o2o_mapping(self):
        """Generates a direct 1-to-1 mapping dictionary."""
        return {
            str(column): MapperRepr(
                f"mapper.val('{column}')", mapper.val(column)
            )
            for column in self.header
            if column
        }

    def process(
        self,
        mapping,
        filename_out,
        params=None,
        t="list",
        null_values=None,
        m2m=False,
    ):
        """Main processor.

        Processes the data using a mapping and prepares it for writing.
        """
        if null_values is None:
            null_values = ["NULL", False]
        if params is None:
            params = {}
        if m2m:
            head, data = self._process_mapping_m2m(
                mapping, null_values=null_values
            )
        else:
            head, data = self._process_mapping(
                mapping, t=t, null_values=null_values
            )

        self._add_data(head, data, filename_out, params)
        return head, data

    def write_to_file(
        self,
        script_filename,
        fail=True,
        append=False,
        python_exe="python",
        path="",
    ):
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
        filename,
        master_key,
        child_key,
        header_prefix="child",
        separator=";",
        encoding="utf-8",
    ):
        """File joiner.

        Joins data from a secondary file into the processor's main data.
        """
        child_header, child_data = self._read_file(
            filename, separator, encoding
        )

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

    def _add_data(self, head, data, filename_out, params):
        params = params.copy()
        params["filename"] = (
            os.path.abspath(filename_out) if filename_out else False
        )
        params["header"] = head
        params["data"] = data
        self.file_to_write[filename_out] = params

    def _process_mapping(self, mapping, t, null_values):
        """The core transformation loop."""
        lines_out = [] if t == "list" else set()
        state = {}  # Persistent state for the entire file processing

        for i, line in enumerate(self.data):
            # Clean up null values
            cleaned_line = [
                s.strip() if s and s.strip() not in null_values else ""
                for s in line
            ]
            line_dict = dict(zip(self.header, cleaned_line))

            try:
                # Pass the state dictionary to each mapper call
                line_out = [
                    mapping[k](line_dict, state) for k in mapping.keys()
                ]
            except SkippingError as e:
                log.debug(f"Skipping line {i}: {e.message}")
                continue
            # This try/except handles mappers that do not accept the `state` dictionary
            # for backward compatibility.
            except TypeError:
                line_out = [mapping[k](line_dict) for k in mapping.keys()]

            if t == "list":
                lines_out.append(line_out)
            else:
                lines_out.add(tuple(line_out))
        return list(mapping.keys()), lines_out

    def _process_mapping_m2m(self, mapping, null_values):
        """m2m process mapping.

        Handles special m2m mapping by expanding list values into unique rows.
        """
        head, data = self._process_mapping(mapping, "list", null_values)
        lines_out = []

        for line_out in data:
            index_list, zip_list = [], []
            for index, value in enumerate(line_out):
                if isinstance(value, list):
                    index_list.append(index)
                    zip_list.append(value)

            if not zip_list:
                # Ensure we don't add duplicate rows
                if line_out not in lines_out:
                    lines_out.append(line_out)
                continue

            # Transpose the lists of values to create new rows
            values_list = zip(*zip_list)
            for values in values_list:
                new_line = list(line_out)
                for i, val in enumerate(values):
                    new_line[index_list[i]] = val

                # Ensure we don't add duplicate rows
                if new_line not in lines_out:
                    lines_out.append(new_line)

        return head, lines_out


class ProductProcessorV10(Processor):
    """Processor to generate a 'product.attribute' file with dynamic variant creation."""

    def process_attribute_data(
        self, attributes_list, ATTRIBUTE_PREFIX, filename_out, import_args
    ):
        """Creates and registers the 'product.attribute.csv' file.

        Args:
            attributes_list (List[str]): list of attribute names (e.g., ['Color', 'Size']).
            ATTRIBUTE_PREFIX (str): Prefix for generating external IDs.
            filename_out (str): Output path for the CSV file.
            import_args (Dict): Arguments for the import script.
        """
        attr_header = ["id", "name", "create_variant"]
        attr_data = [
            [mapper.to_m2o(ATTRIBUTE_PREFIX, att), att, "Dynamically"]
            for att in attributes_list
        ]
        self._add_data(attr_header, attr_data, filename_out, import_args)


class ProductProcessorV9(Processor):
    """Processor to generate variant data from a flat file, creating three CSV files:
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
        mapping: dict,
        attributes_list: list[str],
        processed_rows: list[dict],
    ) -> set[tuple]:
        """Extracts and transforms data for 'product.attribute.value.csv'.

        This replaces the original complex nested 'add_value_line' function.
        """
        attribute_values = set()
        # The 'name' mapping is expected to return a dict of {attribute: value}
        name_key = "name"  # This is a mandatory key in the original mapping

        for row_dict in processed_rows:
            # Apply all mapping functions to the current row
            try:
                line_out_results = [
                    mapping[k](row_dict) for k in mapping.keys()
                ]
            except TypeError:
                line_out_results = [
                    mapping[k](row_dict, {}) for k in mapping.keys()
                ]

            # Find the result of the 'name' mapping, which contains the values
            name_mapping_index = list(mapping.keys()).index(name_key)
            values_dict = line_out_results[name_mapping_index]

            if not isinstance(values_dict, dict):
                continue

            for attr_name in attributes_list:
                # If the attribute exists for this product,
                # create a line for its value
                if values_dict.get(attr_name):
                    value_line = tuple(
                        res[attr_name] if isinstance(res, dict) else res
                        for res in line_out_results
                    )
                    attribute_values.add(value_line)

        return attribute_values

    def process_attribute_mapping(
        self,
        mapping: dict,
        line_mapping: dict,
        attributes_list: list[str],
        ATTRIBUTE_PREFIX: str,
        path: str,
        import_args: dict,
        id_gen_fun: Optional[Callable] = None,
        null_values: Optional[list[str]] = None,
    ):
        """Orchestrates the processing of product attributes and variants from source data."""
        # 1. Generate base attribute data (product.attribute.csv)
        if null_values is None:
            null_values = ["NULL"]
        attr_header, attr_data = self._generate_attribute_file_data(
            attributes_list, ATTRIBUTE_PREFIX
        )

        # 2. Clean and process all data rows into a list of dictionaries
        processed_rows = []
        for line in self.data:
            cleaned_line = [
                s.strip() if s.strip() not in null_values else "" for s in line
            ]
            processed_rows.append(dict(zip(self.header, cleaned_line)))

        # 3. Generate attribute value data (product.attribute.value.csv)
        values_header = list(mapping.keys())
        values_data = self._extract_attribute_value_data(
            mapping, attributes_list, processed_rows
        )

        # 4. Generate attribute line data (product.attribute.line.csv)
        id_gen_fun = id_gen_fun or (
            lambda tmpl_id, vals: mapper.to_m2o(
                tmpl_id.split(".")[0] + "_LINE", tmpl_id
            )
        )
        line_aggregator = AttributeLineDict(attr_data, id_gen_fun)
        for row_dict in processed_rows:
            try:
                values_lines = [
                    line_mapping[k](row_dict) for k in line_mapping.keys()
                ]
            except TypeError:
                values_lines = [
                    line_mapping[k](row_dict, {}) for k in line_mapping.keys()
                ]
            line_aggregator.add_line(values_lines, list(line_mapping.keys()))
        line_header, line_data = line_aggregator.generate_line()

        # 5. Add all three generated files to the write queue
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
