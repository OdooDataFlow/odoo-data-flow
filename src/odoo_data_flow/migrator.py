"""Migrate data between two odoo databases.

This module contains the logic for performing a direct, in-memory
migration of data from one Odoo instance to another.
"""

from typing import Any, Callable, Optional

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
    mapping: Optional[dict[str, Callable[..., Any]]] = None,
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
    )

    if not header or not data:
        log.warning("No data exported. Migration finished.")
        return

    log.info(f"Successfully exported {len(data)} records.")

    # Step 2: Transform the data in memory
    log.info("Transforming data in memory...")
    processor = Processor(header=header, data=data)

    final_mapping: dict[str, Callable[..., Any]]
    if not mapping:
        log.info("No mapping provided, using 1-to-1 mapping.")
        # Convert the MapperRepr dict to a callable dict for the process method
        final_mapping = {k: v.func for k, v in processor.get_o2o_mapping().items()}
    else:
        final_mapping = mapping

    # The process method returns the transformed header and data
    to_import_header, to_import_data_unioned = processor.process(
        final_mapping, filename_out=""
    )

    # Ensure to_import_data is a list of lists
    to_import_data_list: list[list[Any]]
    if isinstance(to_import_data_unioned, set):
        to_import_data_list = [list(row) for row in to_import_data_unioned]
    else:
        to_import_data_list = to_import_data_unioned

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
