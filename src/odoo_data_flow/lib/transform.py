"""This module contains the core Processor class for transforming data."""

import csv
import os
from collections import OrderedDict

from ..logging_config import log
from . import mapper
from .internal.exceptions import SkippingException
from .internal.io import write_file


class MapperRepr:
    """A wrapper to provide a useful string representation for mapper functions."""

    def __init__(self, repr_string, func):
        self._repr_string = repr_string
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
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
            xml_args = {k: v for k, v in kwargs.items() if k.startswith("xml_")}
            self.header, self.data = self._read_file(
                filename, separator, encoding, **xml_args
            )
        elif header is not None and data is not None:
            self.header = header
            self.data = data
        else:
            raise ValueError(
                "Processor must be initialized with either a 'filename' or both 'header' and 'data'."
            )

        # Apply any pre-processing hooks
        self.header, self.data = preprocess(self.header, self.data)

    def _read_file(self, filename, separator, encoding, **kwargs):
        """Reads a CSV file and returns its header and data."""
        # This check should be updated if more file types are supported
        if kwargs.get("xml_root_tag"):
            # Logic for reading XML would go here, for now we assume CSV
            # and that the old xml_transform.py will be removed.
            # This part will need implementation if XML support is kept.
            raise NotImplementedError(
                "XML file processing needs to be integrated into the standard Processor."
            )

        log.info(f"Reading CSV file: {filename}")
        try:
            with open(filename, encoding=encoding, newline="") as f:
                reader = csv.reader(f, separator=separator)
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
        params={},
        t="list",
        null_values=["NULL", False],
        m2m=False,
    ):
        """Processes the data using a mapping and prepares it for writing."""
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
        """Generates the .sh script for the import."""
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
        """Joins data from a secondary file into the processor's main data."""
        child_header, child_data = self._read_file(
            filename, separator, encoding
        )

        try:
            child_key_pos = child_header.index(child_key)
            master_key_pos = self.header.index(master_key)
        except ValueError as e:
            log.error(
                f"Join key error: {e}. Check if '{master_key}' and '{child_key}'"
                f" exist in their respective files."
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
            except SkippingException as e:
                log.debug(f"Skipping line {i}: {e.message}")
                continue

            if t == "list":
                lines_out.append(line_out)
            else:
                lines_out.add(tuple(line_out))
        return list(mapping.keys()), lines_out

    def _process_mapping_m2m(self, mapping, null_values):
        """Handles special m2m mapping by expanding list values into unique rows."""
        head, data = self._process_mapping(mapping, "list", null_values)
        lines_out = set()

        for line_out in data:
            index_list, zip_list = [], []
            for index, value in enumerate(line_out):
                if isinstance(value, list):
                    index_list.append(index)
                    zip_list.append(value)

            if not zip_list:
                lines_out.add(tuple(line_out))
                continue

            # Transpose the lists of values to create new rows
            values_list = zip(*zip_list)
            for values in values_list:
                new_line = list(line_out)
                for i, val in enumerate(values):
                    new_line[index_list[i]] = val
                lines_out.add(tuple(new_line))

        return head, lines_out


class ProductProcessorV9(Processor):
    """Legacy processor for Odoo v9 product imports.
    Note: The `process_attribute_mapping` method is highly specialized
    and a multi-step process using the standard Processor is now preferred.
    """

    def process_attribute_mapping(
        self,
        mapping,
        line_mapping,
        attributes_list,
        ATTRIBUTE_PREFIX,
        path,
        import_args,
        id_gen_fun=None,
        null_values=["NULL"],
    ):
        # ... (original logic would be here, refactored for syntax) ...
        log.warning(
            "Using legacy ProductProcessorV9. "
            "Consider refactoring to the standard Processor."
        )


class ProductProcessorV10(Processor):
    """Processor with helpers for Odoo v10+ product imports."""

    def process_attribute_data(
        self, attributes_list, ATTRIBUTE_PREFIX, filename_out, import_args
    ):
        attr_header = ["id", "name", "create_variant"]
        attr_data = [
            [mapper.to_m2o(ATTRIBUTE_PREFIX, att), att, "Dynamically"]
            for att in attributes_list
        ]
        self._add_data(attr_header, attr_data, filename_out, import_args)
