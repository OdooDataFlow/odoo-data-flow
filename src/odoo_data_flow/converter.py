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
    encoding: str = "utf-8",
) -> None:
    """Path to image.

    Takes a CSV file and converts columns containing local file paths
    into base64 encoded strings.
    """
    log.info(f"Starting path-to-image conversion for file: {file}")
    field_list = fields.split(",")
    processor = Processor(filename=file, separator=delimiter, encoding=encoding)

    mapping = {
        **{
            field: mapper.binary(field, path)
            for field in field_list
            if field in processor.dataframe.columns
        },
        **{
            col: mapper.val(col)
            for col in processor.dataframe.columns
            if col not in field_list
        },
    }
    result_df = processor.process(mapping, filename_out=out)

    # Cast object columns to string before writing to CSV
    for col_name in result_df.columns:
        if result_df[col_name].dtype == pl.Object:
            result_df = result_df.with_columns(
                pl.col(col_name).map_elements(
                    lambda x: str(x) if x is not None else None, return_dtype=pl.String
                )
            )

    result_df.write_csv(out, separator=delimiter)
    log.info(f"Conversion complete. Output written to: {out}")


def run_url_to_image(
    file: str,
    fields: str,
    out: str,
    delimiter: str = ";",
    encoding: str = "utf-8",
    b64: bool = False,
) -> None:
    """URL to image.

    Takes a CSV file and converts columns containing URLs
    into base64 encoded strings by downloading the content.
    """
    log.info(f"Starting URL-to-image conversion for file: {file}")
    field_list = fields.split(",")
    processor = Processor(filename=file, separator=delimiter, encoding=encoding)

    mapping = {
        **{
            field: mapper.binary_url_map(field, b64)
            for field in field_list
            if field in processor.dataframe.columns
        },
        **{
            col: mapper.val(col)
            for col in processor.dataframe.columns
            if col not in field_list
        },
    }

    result_df = processor.process(mapping, filename_out=out)
    # Cast object columns to string before writing to CSV
    for col_name in result_df.columns:
        if result_df[col_name].dtype == pl.Object:
            result_df = result_df.with_columns(
                pl.col(col_name).map_elements(
                    lambda x: str(x) if x is not None else None, return_dtype=pl.String
                )
            )

    result_df.write_csv(out, separator=delimiter)
    log.info(f"Conversion complete. Output written to: {out}")
