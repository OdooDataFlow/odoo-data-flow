"""CSV Data converter.

This module contains functions for converting data, such as image paths
or URLs to base64 strings, for use in Odoo imports.
"""

import base64

import polars as pl

from .lib import mapper
from .lib.transform import Processor
from .logging_config import log


def to_base64(filepath: str) -> str:
    """Reads a local file and returns its base64 encoded content."""
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        log.warning(f"File not found at '{filepath}', skipping.")
        return ""  # Return empty string if file is not found


def run_path_to_image(
    file: str,
    fields: str,
    out: str,
    path: str,
    delimiter: str = ";",
) -> None:
    """Path to image.

    Converts local file paths in specified columns
    into base64 encoded strings.
    """
    log.info(f"Starting path-to-image conversion for file: {file}")
    o2o_mapping = Processor(
        source_filename=file, mapping={}, separator=delimiter
    ).get_o2o_mapping()

    # Create the final mapping, overriding the image path fields
    mapping = {
        **o2o_mapping,
        **{f: mapper.path_to_image(f, path=path) for f in fields.split(",")},
    }

    processor = Processor(
        mapping=mapping,
        source_filename=file,
        separator=delimiter,
    )

    result_df = processor.process(filename_out=out)
    cast_expressions = [
        pl.col(c.name).map_elements(
            lambda x: str(x) if x is not None else "", return_dtype=pl.String
        )
        for c in result_df
        if c.dtype == pl.Object
    ]
    if cast_expressions:
        result_df = result_df.with_columns(cast_expressions)

    result_df.write_csv(out, separator=delimiter)


def run_url_to_image(
    file: str,
    fields: str,
    out: str,
    delimiter: str = ";",
    b64: bool = False,
) -> None:
    """URL to image.

    Downloads images from URLs in specified columns and converts them to base64.
    """
    log.info(f"Starting URL-to-image conversion for file: {file}")
    o2o_mapping = Processor(
        mapping={}, source_filename=file, separator=delimiter
    ).get_o2o_mapping()
    mapping = {
        **o2o_mapping,
        **{f: mapper.binary_url_to_base64(f) for f in fields.split(",")},
    }

    # FIX: Initialize processor with the correct mapping
    processor = Processor(
        mapping=mapping,
        source_filename=file,
        separator=delimiter,
    )

    # FIX: Get the resulting DataFrame and write it directly to a CSV
    result_df = processor.process(filename_out=out)

    cast_expressions = [
        pl.col(c.name).map_elements(
            lambda x: str(x) if x is not None else "", return_dtype=pl.String
        )
        for c in result_df
        if c.dtype == pl.Object
    ]
    if cast_expressions:
        result_df = result_df.with_columns(cast_expressions)

    result_df.write_csv(out, separator=delimiter)
