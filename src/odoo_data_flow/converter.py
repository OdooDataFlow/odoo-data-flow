"""This module contains functions for converting data, such as image paths
or URLs to base64 strings, for use in Odoo imports.
"""

import base64
import os

from .lib import mapper
from .lib.transform import Processor
from .logging_config import log


def to_base64(filepath):
    """Reads a local file and returns its base64 encoded content."""
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        log.warning(f"File not found at '{filepath}', skipping.")
        return ""  # Return empty string if file is not found


def run_path_to_image(file, fields, out="out.csv", path=None):
    """Takes a CSV file and converts columns containing local file paths
    into base64 encoded strings.
    """
    log.info("Starting path-to-image conversion...")

    base_path = path or os.getcwd()

    processor = Processor(file)
    mapping = processor.get_o2o_mapping()

    for f in fields.split(","):
        field_name = f.strip()
        if field_name not in mapping:
            log.warning(
                f"Field '{field_name}' not found in source file. Skipping."
            )
            continue

        log.info(f"Setting up conversion for column: '{field_name}'")
        mapping[field_name] = mapper.val(
            field_name,
            postprocess=lambda x: to_base64(os.path.join(base_path, x))
            if x
            else "",
        )

    processor.process(mapping, out, {}, "list")
    processor.write_to_file("")
    log.info(f"Conversion complete. Output written to '{out}'.")


def run_url_to_image(file, fields, out="out.csv"):
    """Takes a CSV file and converts columns containing URLs
    into base64 encoded strings by downloading the content.
    """
    log.info("Starting url-to-image conversion...")

    processor = Processor(file)
    mapping = processor.get_o2o_mapping()

    for f in fields.split(","):
        field_name = f.strip()
        if field_name not in mapping:
            log.warning(
                f"Field '{field_name}' not found in source file. Skipping."
            )
            continue

        log.info(
            f"Setting up URL download and conversion for column: '{field_name}'"
        )
        # Use the binary_url_map mapper to download and encode the content from the URL
        mapping[field_name] = mapper.binary_url_map(field_name)

    processor.process(mapping, out, {}, "list")
    processor.write_to_file("")
    log.info(f"Conversion complete. Output written to '{out}'.")
