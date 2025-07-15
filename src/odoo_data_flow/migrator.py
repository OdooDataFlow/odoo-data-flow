"""Migrate data between two odoo databases.

This module contains the logic for performing a direct, in-memory
migration of data from one Odoo instance to another.
"""

from collections.abc import Mapping
from typing import Any, Callable, Optional, Union

import polars as pl

from .exporter import run_export_for_migration
from .importer import run_import_for_migration
from .lib.transform import Processor
from .logging_config import log


def run_migration(
    config_export: str,
    config_import: str,
    model: str,
    domain: str = "[]",
    fields: Optional[list[str]] = None,
    mapping: Optional[Mapping[str, Callable[..., Any]]] = None,
    export_worker: int = 1,
    export_batch_size: int = 100,
    import_worker: int = 1,
    import_batch_size: int = 10,
) -> None:
    """Performs a server-to-server data migration.

    This function chains together the export, transform, and import processes
    without creating intermediate files.
    """
    log.info("--- Starting Server-to-Server Migration ---")

    # Step 1: Export data from the source database
    log.info(f"Exporting data from model '{model}'...")
    header, data = run_export_for_migration(
        config=config_export,
        model=model,
        domain=domain,
        fields=fields or [],
        worker=export_worker,
        batch_size=export_batch_size,
        technical_names=True,
    )

    if not header or not data:
        log.warning("No data exported. Migration finished.")
        return

    log.info(f"Successfully exported {len(data)} records.")

    # Step 2: Transform the data in memory
    log.info("Transforming data in memory...")
    df = pl.DataFrame(data, schema=header, orient="row")

    final_mapping: Mapping[str, Union[Callable[..., Any], pl.Expr]]
    if not mapping:
        log.info("No mapping provided, using 1-to-1 mapping.")
        # Create a temporary processor just to generate the o2o map
        temp_processor = Processor(mapping={}, dataframe=df)
        final_mapping = temp_processor.get_o2o_mapping()
    else:
        final_mapping = mapping

    # Create the Processor with the final mapping provided at initialization.
    processor = Processor(mapping=final_mapping, dataframe=df)

    # Call process() with keyword arguments and WITHOUT the mapping.
    result_df = processor.process(filename_out="")

    to_import_header = result_df.columns
    to_import_data_list = [list(row) for row in result_df.rows()]

    # Step 3: Import the transformed data into the destination database
    log.info(f"Importing {len(to_import_data_list)} records into destination...")
    run_import_for_migration(
        config=config_import,
        model=model,
        header=to_import_header,
        data=to_import_data_list,
        worker=import_worker,
        batch_size=import_batch_size,
    )

    log.info("--- Migration Finished Successfully ---")
